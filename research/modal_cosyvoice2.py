"""CosyVoice2 zero-shot inference on Modal. A spike, not part of the harness.

    modal run research/modal_cosyvoice2.py
    modal run research/modal_cosyvoice2.py --text "hello there" --stream

Point of this file: get CosyVoice2 actually running once, see what its API hands
back, and time the first chunk. The harness can then wire it with no surprises.

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


@app.function(gpu="A10G", volumes={"/cache": cache, "/out": out}, timeout=3600)
def synth(text: str, stream: bool, prompt_wav: str, prompt_text: str) -> list[str]:
    import time

    import torchaudio
    from modelscope import snapshot_download

    from cosyvoice.cli.cosyvoice import CosyVoice2

    model_dir = snapshot_download(MODEL, local_dir=f"/cache/{MODEL}")
    cosyvoice = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
    
    # Throwaway pass first. The first inference pays for CUDA kernel selection,
    # lazy module init and the text frontend building its FSTs, which is seconds
    # and has nothing to do with synthesis. Without this the first chunk reads
    # around 8s and the per-chunk RTF falls run-long as the model warms, which
    # is a warmup curve being mistaken for a latency measurement.
    for _ in cosyvoice.inference_cross_lingual("Warming up.", prompt_wav, stream=True):
        pass
    print("warmed", flush=True)

    t0 = time.monotonic()
    paths = []
    # The reference goes in as a path, not a loaded tensor: the frontend wants it
    # at both 16k and 24k and does its own reading.
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
        print(f"chunk {i}: {(time.monotonic() - t0) * 1000:7.0f} ms, {dur:.2f}s audio",
              flush=True)
        path = f"/out/cosyvoice2_{i:03d}.wav"
        torchaudio.save(path, audio, cosyvoice.sample_rate)
        paths.append(path)

    print(f"total {(time.monotonic() - t0) * 1000:.0f} ms, {len(paths)} chunks")
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
):
    print("\n".join(synth.remote(text, stream, prompt_wav, prompt_text)))
