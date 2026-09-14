"""Runs the whole sweep on Modal as one job.

    modal run modal_app.py                       # everything
    modal run modal_app.py --configs cascade_batch_f5,moshi

A benchmark is a job, not a service: one long invocation that sets up, runs
every config, writes to durable storage and exits. That plays to Modal's
strengths, because the cold start is paid once before any measured trial and the
container's disposability stops mattering.

One container with several GPUs, not one function per stage. Separate functions
land in separate containers on separate machines, so everything the stages
exchange would cross a network, and for a pipeline measured in milliseconds that
hop is not a detail, it is the measurement.

Four things that bite, all handled below: committing the Volume (without it the
traces are written and then gone), pointing HF_HOME at a Volume (or every run
re-downloads every checkpoint), the default timeout (the feeder streams at 1x,
so a sweep takes wall-clock hours), and pinning the same GPU class for every
reported run.
"""

import subprocess
import sys

import modal

GPU = "A10G:3"          # one container, three cards: llm, asr, tts
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

app = modal.App("s2s-latency-bench")

models = modal.Volume.from_name("s2s-models", create_if_missing=True)
results = modal.Volume.from_name("s2s-results", create_if_missing=True)

image = (
    modal.Image.from_dockerfile("Dockerfile")
    # Weights land on a Volume so they download once rather than every run.
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    # Mounted, not baked. Baking the repo into a layer means every commit
    # invalidates it and you wait through a rebuild to test a one-line change.
    .add_local_dir("bench", "/root/bench")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("data", "/root/data")
)


@app.function(
    gpu=GPU,
    image=image,
    volumes={"/cache": models, "/results": results},
    timeout=TIMEOUT_S,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_bench(configs: list[str]) -> list[str]:
    written = []
    for name in configs:
        out = f"/results/{name}.jsonl"
        print(f"=== {name}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "bench.runner",
             "--config", f"configs/{name}.yaml", "--out", out],
            cwd="/root", check=True,
        )
        written.append(out)
        # Commit per config rather than at the end, so a crash halfway through
        # keeps the configs that already finished.
        results.commit()
    return written


@app.function(image=image, volumes={"/results": results}, timeout=600)
def report() -> str:
    out = subprocess.run(
        [sys.executable, "scripts/report.py", "/results/*.jsonl"],
        cwd="/root", check=True, capture_output=True, text=True,
    )
    return out.stdout


@app.local_entrypoint()
def main(configs: str = ""):
    names = [c.strip() for c in configs.split(",") if c.strip()] or ALL_CONFIGS
    unknown = set(names) - set(ALL_CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}")
    print("\n".join(run_bench.remote(names)))
    print(report.remote())
