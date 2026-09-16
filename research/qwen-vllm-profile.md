# Profiling Qwen3-4B on vLLM

What the LLM stage costs, measured before wiring it into the harness.

One A10G on Modal, `Qwen/Qwen3-4B-Instruct-2507`, bf16, temperature 0, seed 0.
Prompt is "What is the capital of France?" with a two-sentence cap. Script is
`modal_vllm_qwen.py`.

```bash
modal run research/modal_vllm_qwen.py --warmup --concurrent 8
```

## Numbers

All times in ms. `2nd tok` is first token to second, which is one decode step
with a warm KV cache and the only clean read on per-token cost, since the first
token carries prefill with it.

**One request**

| | ttft | 2nd tok | decode | total | steps | tok/s |
|---|---|---|---|---|---|---|
| no warmup | 73 | 20.3 | 126 | 200 | 7 | 47.6 |
| warmup | 43 | 20.1 | 127 | 169 | 7 | 47.3 |

**Concurrency, all warm**

| | ttft | 2nd tok | decode | total | steps | tok/s |
|---|---|---|---|---|---|---|
| 1 | 43 | 20.1 | 127 | 169 | 7 | 47.3 |
| 2 | 42, 62 | 20.4 | 123, 124 | 165, 186 | 7 | 48.4 |
| 4 | 43, 65, 65, 66 | 20.6 | 129 | 173 – 195 | 7 | 46.6 |
| 8 | 47, 67 – 68 | 20.3 | 124 – 125 | 171 – 192 | 7 | 48.5 |

## What it shows

**Continuous batching holds flat.** Eight concurrent requests cost the same per
token as one: 20.3 ms either way, 47 to 48 tok/s each. Nothing degrades across
the range the harness sweeps.

**ttft barely moves.** 43 ms at concurrency 1, 68 ms at 8. The spread inside a
run is submission order, not contention, since requests join the batch as they
arrive and the first one in gets its prefill first.

**Output is deterministic.** Every request produced exactly 7 steps, at every
concurrency level. That is the property `check_work_constant` depends on, and the
LLM has it. CosyVoice2 does not, so the TTS is the side that needs pinning.

**Warmup is worth about 30 ms**, all of it on ttft. Worth doing, but nothing like
CosyVoice2, where it was worth 2.4 seconds.

## Caveats

The load is small. Seven tokens per request against a 96 token cap, and a short
prompt, so the GPU is nowhere near saturated. Flat scaling to 8 is expected here
and says nothing about where the knee is. Rerun with longer prompts and real
responses before trusting it as a concurrency result.

`enforce_eager=True` is set, which skips CUDA graph capture, so 20.3 ms per token
is slower than the harness will actually get. It was on to isolate a startup
failure and should come off now that startup is reliable.
