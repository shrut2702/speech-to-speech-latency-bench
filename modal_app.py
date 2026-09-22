"""Runs the sweep on Modal as one job.

    modal run modal_app.py                                  # everything
    modal run modal_app.py --configs cascade_batch_cosyvoice2

A benchmark is a job, not a service: one long invocation that sets up, runs
every config, writes to durable storage and exits. The cold start is paid once,
before any measured trial, and the container's disposability stops mattering.

One container with three GPUs, not a function per stage. Separate functions land
in separate containers on separate machines, so everything the stages exchange
would cross a network, and for a pipeline measured in milliseconds that hop is
not a detail, it is the measurement. Inside the container each stage still gets
its own process and its own card, which the harness handles.

Results go to a folder per config on the volume, committed after each one so a
crash keeps what already finished.
"""

import subprocess
import sys

import modal

# One card per stage. A100 40GB rather than A10G: Moshi needs ~16GB in bf16 and
# would not fit otherwise, and pinning the same class across every reported run
# matters more than picking the cheapest one that fits this config.
GPU = "A100-40GB:3"   # asr, llm, tts
TIMEOUT_S = 12 * 3600

ALL_CONFIGS = [
    "cascade_batch_cosyvoice2",
    "cascade_stream_gen_cosyvoice2",
    "cascade_stream_all_cosyvoice2",
    "cascade_batch_f5",
    "cascade_stream_gen_f5",
    "cascade_stream_all_f5",
    "moshi",
]

COSYVOICE = "/opt/CosyVoice"
WHISPER_STREAMING = "/opt/whisper_streaming"

app = modal.App("s2s-latency-bench")

models = modal.Volume.from_name("s2s-models", create_if_missing=True)
results = modal.Volume.from_name("s2s-results", create_if_missing=True)

TTS_VENV = "/opt/tts-venv"

image = (
    modal.Image.micromamba(python_version="3.10")
    # pynini via conda-forge, for CosyVoice's text frontend. pip builds of it
    # fail on glibc. Installed globally because it's a native extension that
    # the TTS venv will inherit via --system-site-packages.
    .micromamba_install("pynini=2.1.5", channels=["conda-forge"])
    .apt_install("git", "ffmpeg", "sox", "libsox-dev", "build-essential")
    .run_commands(
        f"git clone --recursive https://github.com/FunAudioLLM/CosyVoice {COSYVOICE}",
        f"git clone --depth 1 https://github.com/ufal/whisper_streaming {WHISPER_STREAMING}",
    )
    # ---- TTS venv (CosyVoice + numpy<2, isolated from vLLM) ----------------
    # A dedicated venv lets CosyVoice pin numpy<2 without fighting vLLM's
    # numpy>=2 requirement. We use a fully isolated venv and link pynini
    # from the base conda environment to avoid leaking base packages like
    # torchvision and vllm's numpy 2 into the TTS worker.
    .run_commands(
        f"python -m venv {TTS_VENV}",
        "echo 'setuptools<81' > /tmp/constraint.txt",
        "echo 'numpy<2' >> /tmp/constraint.txt",
        f"{TTS_VENV}/bin/pip install 'setuptools<81' wheel",
        f"PIP_CONSTRAINT=/tmp/constraint.txt {TTS_VENV}/bin/pip install -r {COSYVOICE}/requirements.txt",
        # deepspeed is a training dep; importing it compiles CUDA ops which
        # needs nvcc this image doesn't carry.
        f"{TTS_VENV}/bin/pip uninstall -y deepspeed || true",
        # Copy pynini from conda base. Done after pip install to ensure site-packages exists.
        f"cp -R /opt/conda/lib/python3.10/site-packages/pynini* {TTS_VENV}/lib/python3.10/site-packages/ || true",
        f"cp -R /opt/conda/lib/python3.10/site-packages/pywrapfst* {TTS_VENV}/lib/python3.10/site-packages/ || true",
    )
    # ---- Global env (ASR + LLM, no numpy constraint) -----------------------
    .pip_install(
        "faster-whisper>=1.0", "vllm>=0.6", "transformers>=4.51",
        "modelscope", "soundfile>=0.12", "soxr>=0.5", "pyyaml>=6.0",
    )
    # CosyVoice's requirements and vLLM's pull different nvidia-cudnn-cu12
    # versions, leaving two libcudnn.so.x side by side. Nothing notices until
    # vLLM probes FlashInfer while choosing an attention backend, whose import
    # chain loads cudnn and asserts on finding two. FlashInfer is optional here:
    # sampling is greedy and A100 has FlashAttention.
    .run_commands(
        "pip uninstall -y flashinfer-python flashinfer || true",
        # CTranslate2, under faster-whisper, dlopens libcublas.so.12 and
        # libcudnn.so.9 by exact soname. CosyVoice's requirements used to pin
        # torch to a CUDA 12 build and supplied them as a side effect; with
        # CosyVoice moved into its own venv nothing constrains the base env, so
        # vLLM pulls a CUDA 13 torch and only .so.13 is present. Install the 12
        # runtimes explicitly rather than dragging torch back a major version.
        "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12",
        # Printed into the build log: if the soname is still wrong this says so
        # at build time instead of at the ASR's first transcription.
        "ls /opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib "
        "/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib",
    )
    .env({
        # Weights on a Volume, so they download once rather than every run.
        "HF_HOME": "/cache/hf",
        "MODELSCOPE_CACHE": "/cache/modelscope",
        "PYTHONUNBUFFERED": "1",
        # whisper_streaming is a repo on PYTHONPATH, not a package.
        # CosyVoice lives in the TTS venv; the harness (ASR/LLM) does not need
        # its path, but runner.py imports stages so we include it for the main
        # process too (harmless: the global env just won't have cosyvoice
        # importable, but _serve inside the TTS venv will).
        "PYTHONPATH": (
            f"/root:{COSYVOICE}:{COSYVOICE}/third_party/Matcha-TTS:{WHISPER_STREAMING}"
        ),
        # FlashInfer JIT-compiles its sampling kernels and needs nvcc. At
        # temperature 0 the native sampler does the same work.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Name the attention backend rather than letting vLLM enumerate them,
        # since enumerating is what imports FlashInfer in the first place.
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        # CTranslate2, under faster-whisper, dlopens libcublas and libcudnn at
        # runtime. torch ships them inside site-packages rather than on the
        # default search path, so without this the ASR loads fine and then
        # fails on its first transcription.
        "LD_LIBRARY_PATH": (
            "/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:"
            "/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib:"
            "/opt/conda/lib"
        ),
    })
    # Mounted, not baked. Baking the repo in means every commit invalidates the
    # layer and you wait through a rebuild to test a one-line change.
    .add_local_dir("bench", "/root/bench")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("data", "/root/data")
    # The TTS reference voice. Mounted rather than baked so swapping speakers
    # does not rebuild the image.
    .add_local_dir("assets", "/root/assets")
)


@app.function(
    gpu=GPU,
    image=image,
    volumes={"/cache": models, "/results": results},
    timeout=TIMEOUT_S,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_bench(
    configs: list[str],
    trials: int = 0,
    max_clips: int = 0,
) -> list[str]:
    written = []
    for name in configs:
        print(f"\n=== {name}", flush=True)
        cmd = [
            sys.executable, "-m", "bench.runner",
            "--config", f"configs/{name}.yaml",
            "--results-dir", "/results",
        ]
        if trials > 0:
            cmd.extend(["--trials", str(trials)])
        if max_clips > 0:
            cmd.extend(["--max-clips", str(max_clips)])
        subprocess.run(cmd, cwd="/root", check=True)
        written.append(f"/results/{name}")
        # Commit per config rather than at the end, so a crash halfway through
        # keeps the configs that already finished.
        results.commit()
    return written


@app.function(image=image, volumes={"/results": results}, timeout=600)
def report(configs: list[str]) -> str:
    traces = [f"/results/{name}/traces.jsonl" for name in configs]
    out = subprocess.run(
        [sys.executable, "scripts/report.py", *traces],
        cwd="/root", check=True, capture_output=True, text=True,
    )
    return out.stdout


@app.local_entrypoint()
def main(
    configs: str = "",
    trials: int = 0,
    max_clips: int = 0,
    sample_only: bool = False,
):
    """Runs the benchmark suite on Modal.

    Usage:
        modal run modal_app.py --configs cascade_batch_cosyvoice2 --sample-only
        modal run modal_app.py --configs cascade_batch_cosyvoice2 --trials 1 --max-clips 1
    """
    if sample_only:
        trials = 1
        max_clips = 1
    names = [c.strip() for c in configs.split(",") if c.strip()] or ALL_CONFIGS
    unknown = set(names) - set(ALL_CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}")
    print("\n".join(run_bench.remote(names, trials=trials, max_clips=max_clips)))
    print(report.remote(names))
