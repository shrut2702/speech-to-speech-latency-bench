"""CosyVoice2 inference on Modal. A spike, not part of the harness.

    modal run research/modal_cosyvoice2.py --stream
    modal run research/modal_cosyvoice2.py --stream --warmup
    modal run research/modal_cosyvoice2.py --stream --warmup --split

Point of this file: get CosyVoice2 running, time the LM and the decoder apart,
and see what a second GPU is worth. Findings live in cosyvoice2-profile.md.

Install notes, because they are the only hard part:
  - CosyVoice is a repo, not a pip package, and needs its Matcha-TTS submodule
    on PYTHONPATH.
  - WeTextProcessing pulls in pynini, which does not pip install cleanly. The
    micromamba image is the shortest path to a working one.
"""

import modal

MODEL = "iic/CosyVoice2-0.5B"
REPO = "/opt/CosyVoice"

image = (
    modal.Image.micromamba(python_version="3.10")
    # pynini via conda-forge. pip builds of it fail on glibc and cost an hour.
    .micromamba_install("pynini=2.1.5", channels=["conda-forge"])
    .apt_install("git", "ffmpeg", "sox", "libsox-dev", "build-essential")
    .run_commands(
        f"git clone --recursive https://github.com/FunAudioLLM/CosyVoice {REPO}",
        # setuptools 81 dropped pkg_resources and one of CosyVoice's
        # requirements still imports it from its setup.py. PIP_CONSTRAINT is the
        # pin that reaches inside pip's isolated build environment; installing
        # an older setuptools alongside does not, because the build env builds
        # its own.
        "echo 'setuptools<81' > /tmp/constraint.txt",
        "pip install 'setuptools<81' wheel",
        f"PIP_CONSTRAINT=/tmp/constraint.txt pip install -r {REPO}/requirements.txt",
        # DeepSpeed is a training dependency. transformers imports it whenever
        # it is installed, and importing it compiles CUDA ops, which needs nvcc
        # that this image does not carry. Inference never touches it.
        "pip uninstall -y deepspeed",
    )
    .pip_install("modelscope")
    .env({"PYTHONPATH": f"{REPO}:{REPO}/third_party/Matcha-TTS"})
)

app = modal.App("cosyvoice2-spike", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)
out = modal.Volume.from_name("s2s-results", create_if_missing=True)


def instrument(cosyvoice, events: list) -> None:
    """Times the LM and the decoder apart.

    Streaming runs the LM in a background thread while the main thread turns
    already-emitted tokens into audio, so the two overlap on one GPU. The public
    API hands back finished audio and hides which of them the time went to.

      model.llm.inference  text -> speech tokens, autoregressive
      model.token2wav      speech tokens -> mel -> waveform, flow + vocoder
    """
    import time

    model = cosyvoice.model
    assert hasattr(model, "token2wav"), f"no token2wav on {type(model).__name__}"

    original_llm = model.llm.inference
    original_token2wav = model.token2wav

    def timed_llm(*args, **kwargs):
        n = 0
        for token in original_llm(*args, **kwargs):
            n += 1
            if n == 1:
                events.append(("llm first token", time.monotonic()))
            yield token
        events.append((f"llm last token ({n})", time.monotonic()))

    def timed_token2wav(*args, **kwargs):
        events.append(("decoder start", time.monotonic()))
        try:
            return original_token2wav(*args, **kwargs)
        finally:
            events.append(("decoder end", time.monotonic()))

    model.llm.inference = timed_llm
    model.token2wav = timed_token2wav


def split_devices(cosyvoice, decoder_device: str = "cuda:1") -> None:
    """Puts the flow decoder and vocoder on a second card.

    The LM and the decoder overlap during streaming, so they contend for the
    same SMs. This measures what separating them is worth rather than inferring
    it from the ~5% gap between a contended decoder pass and the uncontended
    final one.

    The awkward part: CosyVoice2Model keeps a single `self.device` and both the
    LM thread and token2wav read it. Swapping it around a call would race, since
    the LM keeps generating while token2wav runs. So `device` becomes
    thread-local. The main thread, which runs token2wav, sees the decoder card;
    the background LM thread keeps seeing the original.
    """
    import threading

    import torch

    model = cosyvoice.model
    base = torch.device(model.device)
    decoder = torch.device(decoder_device)

    model.flow.to(decoder)
    model.hift.to(decoder)

    local = threading.local()
    # A property on the class is a data descriptor, so it takes precedence over
    # the instance attribute that is already set.
    type(model).device = property(lambda self: getattr(local, "device", base))

    original = model.token2wav

    def on_decoder_card(*args, **kwargs):
        local.device = decoder
        try:
            return original(*args, **kwargs)
        finally:
            local.device = base

    model.token2wav = on_decoder_card
    print(f"llm on {base}, decoder on {decoder}", flush=True)


# Two cards so the split can be measured. Without --split the second one idles,
# which keeps both arms of the comparison on one image.
@app.function(gpu="A10G:2", volumes={"/cache": cache, "/out": out}, timeout=3600)
def synth(
    text: str, stream: bool, prompt_wav: str, prompt_text: str,
    warmup: bool, split: bool,
) -> list[str]:
    import time

    import torchaudio
    from modelscope import snapshot_download

    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_dir = snapshot_download(MODEL, local_dir=f"/cache/{MODEL}")
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

    if split:
        split_devices(cosyvoice)

    if warmup:
        # The first inference pays for CUDA kernel selection, lazy module init
        # and the text frontend building its FSTs. Seconds, and nothing to do
        # with synthesis. Without it the first chunk reads around 4s instead of
        # 1.7s and the per-chunk RTF falls run-long, which is a warmup curve
        # being mistaken for a latency measurement.
        for _ in cosyvoice.inference_cross_lingual(
            "Warming up.", prompt_wav, stream=True
        ):
            pass
        print("warmed", flush=True)

    events: list = []
    instrument(cosyvoice, events)

    t0 = time.monotonic()
    paths = []
    # The reference goes in as a path, not a loaded tensor: the frontend wants
    # it at both 16k and 24k and does its own reading.
    #
    # Zero-shot clones the reference voice and needs its transcript.
    # Cross-lingual is the same clone with the target in another language, so it
    # takes no transcript. The repo's shipped prompt is a Chinese speaker, so
    # English text through it lands here and comes out Chinese-accented. For the
    # benchmark, pass an English reference instead and the zero-shot path runs.
    gen = (
        cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav, stream=stream)
        if prompt_text
        else cosyvoice.inference_cross_lingual(text, prompt_wav, stream=stream)
    )
    for i, chunk in enumerate(gen):
        audio = chunk["tts_speech"]
        dur = audio.shape[-1] / cosyvoice.sample_rate
        events.append((f"chunk {i} out ({dur:.2f}s audio)", time.monotonic()))
        path = f"/out/cosyvoice2_{i:03d}.wav"
        torchaudio.save(path, audio, cosyvoice.sample_rate)
        paths.append(path)

    total = (time.monotonic() - t0) * 1000
    # Printed at the end, not as they happen. The LM runs on another thread, so
    # inline prints would interleave the two and misrepresent the order.
    print("\n--- timeline, ms from synthesis start ---", flush=True)
    previous = t0
    for label, when in sorted(events, key=lambda e: e[1]):
        print(f"{(when - t0) * 1000:8.0f}  (+{(when - previous) * 1000:6.0f})  {label}")
        previous = when
    print(f"total {total:.0f} ms, {len(paths)} chunks", flush=True)

    out.commit()
    return paths


@app.local_entrypoint()
def main(
    # Three sentences, so streaming has something to chunk. One sentence yields
    # a single piece and tells you nothing about how the chunks arrive.
    text: str = (
        "The capital of France is Paris. "
        "It has been the seat of government since the tenth century. "
        "Today it is home to just over two million people."
    ),
    stream: bool = False,
    # Defaults to the repo's own asset so this runs with no setup. That speaker
    # is Chinese, so English text goes through the cross-lingual path and picks
    # up the accent. Pass an English wav plus its transcript for a clean voice.
    prompt_wav: str = f"{REPO}/asset/zero_shot_prompt.wav",
    prompt_text: str = "",
    warmup: bool = False,
    # Flow decoder and vocoder onto cuda:1, LM stays on cuda:0.
    split: bool = False,
):
    print("\n".join(synth.remote(text, stream, prompt_wav, prompt_text, warmup, split)))
