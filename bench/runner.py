"""Runs a config over the clip set and writes traces.

    python -m bench.runner --config configs/cascade_batch.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .feeder import AudioFeeder
from .systems.base import S2SSystem
from .trace import Trace, TraceWriter


def load_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def save_audio(root: Path, trace: Trace, chunks: list) -> None:
    """Writes one trial's output audio, one wav per chunk plus the whole thing.

    The per-chunk files are the point for the streaming paths: they show what
    arrived when, which is what the inter-chunk gaps in the report are measuring.
    Batch produces a single chunk, so the two files are the same audio.
    """
    if not chunks:
        return
    out = root / trace.config / trace.clip_id / f"trial{trace.trial}"
    out.mkdir(parents=True, exist_ok=True)
    sr = chunks[0].sample_rate
    for i, c in enumerate(chunks):
        sf.write(out / f"chunk_{i:03d}.wav", c.samples, c.sample_rate)
    sf.write(out / "full.wav", np.concatenate([c.samples for c in chunks]), sr)


def build_system(cfg: dict) -> S2SSystem:
    kind = cfg["system"]
    if kind == "cascade":
        from .systems.cascade import CascadeSystem

        return CascadeSystem(cfg)
    if kind == "moshi":
        from .systems.moshi import MoshiSystem

        return MoshiSystem(cfg)
    raise ValueError(f"unknown system: {kind}")


async def run_config(cfg: dict, out_path: Path) -> None:
    manifest = load_manifest(cfg["manifest"])
    clips_dir = Path(cfg.get("clips_dir", "data/clips"))
    trials = int(cfg.get("trials", 20))
    frame_ms = int(cfg.get("frame_ms", 20))
    silence_after_s = float(cfg.get("silence_after_s", 20.0))

    # Recorded into every trace so numbers from different machines can never be
    # silently compared.
    env = {
        "gpu": gpu_name(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": cfg,
    }

    system = build_system(cfg)
    await system.load()

    def make_feeder(row: dict) -> AudioFeeder:
        samples, sr = sf.read(clips_dir / row["file"], dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return AudioFeeder(
            samples=samples,
            sample_rate=sr,
            endpoint_sample=row["endpoint_sample"],
            frame_ms=frame_ms,
            # The mic does not switch off while the user waits for an answer.
            # Full-duplex models need these frames to keep generating; the
            # cascade ignores them. Must exceed the longest expected response.
            silence_after_s=silence_after_s,
        )

    await system.warmup(lambda: make_feeder(manifest[0]))

    # Optional. On Modal point this at the results Volume so the audio survives
    # the container. Off by default because a full sweep is a lot of wav files.
    audio_dir = Path(cfg["audio_dir"]) if cfg.get("audio_dir") else None

    writer = TraceWriter(out_path)
    for row in manifest:
        for trial in range(trials):
            trace = Trace(
                clip_id=row["id"],
                config=cfg["name"],
                trial=trial,
                env=env,
            )
            produced: list = []
            async for chunk in system.run(make_feeder(row), trace):
                if audio_dir is not None:
                    produced.append(chunk)
            # Written after the trial, never during it. Synthesis is being timed
            # to the millisecond and a disk write inside the loop would land in
            # the inter-chunk gaps.
            if audio_dir is not None:
                save_audio(audio_dir, trace, produced)
            writer.write(trace)
            print(f"{cfg['name']} {row['id']} trial {trial + 1}/{trials}")

    await system.unload()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="defaults to results/<name>.jsonl")
    ap.add_argument("--audio-dir", default=None,
                    help="save output audio here; overrides audio_dir in the config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.audio_dir:
        cfg["audio_dir"] = args.audio_dir
    out = Path(args.out) if args.out else Path("results") / f"{cfg['name']}.jsonl"
    asyncio.run(run_config(cfg, out))


if __name__ == "__main__":
    main()
