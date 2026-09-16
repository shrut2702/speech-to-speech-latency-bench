"""Qwen3-4B on vLLM on Modal. A spike, not part of the harness.

    modal run research/modal_vllm_qwen.py
    modal run research/modal_vllm_qwen.py --warmup
    modal run research/modal_vllm_qwen.py --warmup --concurrent 4

Point of this file: see what the LLM stage actually costs before wiring it in.
Three numbers matter and they come from different parts of the model's work:

  ttft            prompt in to first token out. Dominated by prefill.
  second token    first token to second. One decode step with a warm KV cache,
                  which is the only honest read on per-token cost, since the
                  first token carries prefill with it.
  decode          first token to last. Prefill excluded, so tok/s divides by it.
  total           request in to last token, which is what llm_total_ms in the
                  harness measures and what a caller actually waits through.

Concurrency is the other reason vLLM is here rather than transformers. With
--concurrent N it fires N requests at once and reports each one's ttft, which is
what continuous batching either degrades gracefully or does not.

Generation is pinned the way the harness pins it: temperature 0, fixed seed,
capped length. If response length drifts, the downstream TTS does a different
amount of work and nothing is comparable.
"""

import modal

# Same string the harness configs use. The -2507 suffix matters: Qwen publishes
# several 4B variants and they are not interchangeable.
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM_PROMPT = "Answer in at most two sentences."
MAX_TOKENS = 96

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.6", "transformers>=4.51", "huggingface_hub[hf_transfer]")
    .env({
        # Weights land on a Volume so they download once rather than every run.
        "HF_HOME": "/cache/hf",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # FlashInfer JIT-compiles its sampling kernels on first use and needs
        # nvcc, which this image doesn't carry. vLLM falls back to its native
        # PyTorch sampler, and at temperature 0 the top-k/top-p path it was
        # building isn't doing anything anyway.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # AsyncLLM runs its engine core in a separate process, so when that
        # process dies you get "Engine core initialization failed" with an empty
        # proc list and the real traceback goes to the core's own stderr. Keep
        # the logging up so it lands in the Modal logs.
        "VLLM_LOGGING_LEVEL": "INFO",
    })
)

app = modal.App("qwen-vllm-spike", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)


@app.function(
    gpu="A10G",
    volumes={"/cache": cache},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate(prompt: str, warmup: bool, concurrent: int) -> None:
    import asyncio

    asyncio.run(_run(prompt, warmup, concurrent))


async def _run(prompt: str, warmup: bool, concurrent: int) -> None:
    import asyncio
    import time
    import uuid

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    # Fetch the weights before the engine starts. The engine core runs in its
    # own process and the client waits on a startup handshake; if that process
    # spends the window pulling 8GB from HuggingFace, startup times out and the
    # failure reports an empty proc list, because nothing actually crashed.
    print(f"fetching {MODEL}", flush=True)
    path = snapshot_download(MODEL)
    print(f"weights at {path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=MODEL,
            dtype="bfloat16",
            max_model_len=2048,
            # 0.90 of a 24GB A10G leaves little headroom for the profiling run
            # and graph capture on top of ~8GB of weights.
            gpu_memory_utilization=0.80,
            # Skips CUDA graph capture. Costs some decode speed, removes a
            # common startup failure. Turn it off once init is reliable, since
            # graphs are worth real milliseconds per token.
            enforce_eager=True,
            disable_log_stats=True,
        )
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Qwen3 emits a reasoning block unless this is off, which would blow past
        # max_tokens before a single spoken word appears.
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)

    sampling = SamplingParams(temperature=0.0, seed=0, max_tokens=MAX_TOKENS)

    async def one(label: str) -> dict:
        """Streams one request, recording when each token arrived."""
        t0 = time.monotonic()
        arrivals: list[float] = []
        emitted = 0
        out_text = ""
        async for output in engine.generate(text, sampling, str(uuid.uuid4())):
            # vLLM hands back cumulative text each step, not the delta.
            out_text = output.outputs[0].text
            if len(out_text) > emitted:
                arrivals.append(time.monotonic())
                emitted = len(out_text)
        return {
            "label": label,
            "ttft_ms": (arrivals[0] - t0) * 1000 if arrivals else None,
            "second_ms": (arrivals[1] - arrivals[0]) * 1000 if len(arrivals) > 1 else None,
            # From the request going in, matching llm_total_ms in the harness.
            "total_ms": (arrivals[-1] - t0) * 1000 if arrivals else None,
            # First token to last. Prefill excluded, so this is the one to
            # divide by for a per-token rate.
            "decode_ms": (arrivals[-1] - arrivals[0]) * 1000 if arrivals else None,
            "steps": len(arrivals),
            "text": out_text,
        }

    if warmup:
        # The first request pays for CUDA graph capture and allocator warmup.
        # Seconds, and nothing to do with generation.
        await one("warmup")
        print("warmed", flush=True)

    results = await asyncio.gather(*[one(f"req {i}") for i in range(concurrent)])

    print(f"\n--- {concurrent} concurrent request(s), warmup={warmup} ---", flush=True)
    print(f"{'':8s} {'ttft':>9s} {'2nd tok':>9s} {'decode':>9s} {'total':>9s} "
          f"{'steps':>6s} {'tok/s':>7s}")
    for r in results:
        rate = (r["steps"] - 1) / (r["decode_ms"] / 1000) if r["decode_ms"] else 0
        print(f"{r['label']:8s} {r['ttft_ms']:9.0f} {r['second_ms']:9.1f} "
              f"{r['decode_ms']:9.0f} {r['total_ms']:9.0f} {r['steps']:6d} {rate:7.1f}")

    print(f"\nresponse: {results[0]['text']!r}")
    # The harness gates on this: if length drifts between configs, the TTS does
    # different work and the latencies stop being comparable.
    lengths = {r["steps"] for r in results}
    print(f"steps across requests: {sorted(lengths)}")


@app.local_entrypoint()
def main(
    prompt: str = "What is the capital of France?",
    warmup: bool = False,
    # The harness sweeps 1, 2, 4, 8. Continuous batching is what decides whether
    # ttft degrades gracefully across that range or falls off a cliff.
    concurrent: int = 8,
):
    generate.remote(prompt, warmup, concurrent)
