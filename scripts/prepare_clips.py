"""Turns raw clips into a comparable test set.

Four steps, all of which matter:

  resample        one sample rate everywhere
  trim            cut to the last speech sample, so trailing silence is uniform
  pad             append exactly `--tail-ms` of silence
  loudness-norm   VAD-free pipelines still have level-sensitive stages, and
                  consistent loudness keeps ASR behaviour stable across clips

It then writes `endpoint_sample`, the annotated end of speech, into the
manifest. That number is t=0 for every latency metric in the benchmark, so it
has to come from the audio rather than from any model's opinion about it.

    python scripts/prepare_clips.py --src raw/ --out data/clips \
        --manifest data/manifest.jsonl --refs raw/refs.json

`--refs` is an optional id -> metadata sidecar (fetch_clips.py writes one). Its
keys are merged into each manifest row, so a clip carries both the transcript of
what was asked and the reference answer. Neither is used for timing, but without
them there is no way to tell whether a latency win cost accuracy, and error
propagation is the cascade's main structural weakness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim > 1 else x


def last_speech_sample(
    x: np.ndarray, sr: int, win_ms: float = 20.0, rel_db: float = -35.0
) -> int:
    """Last frame whose energy is within `rel_db` of the clip's peak frame.

    Deliberately simple and deterministic. This is annotation, not detection:
    it runs once offline and the answer is committed to the manifest, so every
    run of every config sees the identical endpoint.
    """
    win = max(1, int(sr * win_ms / 1000))
    n = len(x) // win
    if n == 0:
        return len(x)
    frames = x[: n * win].reshape(n, win)
    energy = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    peak = energy.max()
    thresh = peak * (10 ** (rel_db / 20))
    above = np.nonzero(energy > thresh)[0]
    if len(above) == 0:
        return len(x)
    return int((above[-1] + 1) * win)


def normalize_rms(x: np.ndarray, target_dbfs: float = -23.0) -> np.ndarray:
    rms = np.sqrt((x.astype(np.float64) ** 2).mean() + 1e-12)
    gain = (10 ** (target_dbfs / 20)) / rms
    y = x * gain
    peak = np.abs(y).max()
    if peak > 0.99:  # avoid clipping
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory of raw wav/flac")
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--tail-ms", type=int, default=800)
    ap.add_argument("--target-dbfs", type=float, default=-23.0)
    ap.add_argument("--refs", default=None,
                    help="optional JSON mapping clip id -> metadata merged into the row")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    refs: dict[str, dict] = {}
    if args.refs:
        raw_refs = json.loads(Path(args.refs).read_text(encoding="utf-8"))
        # Tolerate a plain id -> transcript mapping as well as full metadata.
        refs = {
            k: ({"transcript": v} if isinstance(v, str) else dict(v))
            for k, v in raw_refs.items()
        }
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".mp3"})
    if not files:
        raise SystemExit(f"no audio found under {src}")

    rows = []
    for p in files:
        x, sr = sf.read(p, dtype="float32")
        x = to_mono(x)

        if sr != args.sr:
            try:
                import soxr

                x = soxr.resample(x, sr, args.sr).astype(np.float32)
            except ImportError:
                raise SystemExit("pip install soxr, or pre-resample your clips")
            sr = args.sr

        end = last_speech_sample(x, sr)
        x = x[:end]
        x = normalize_rms(x, args.target_dbfs)

        tail = np.zeros(int(sr * args.tail_ms / 1000), dtype=np.float32)
        y = np.concatenate([x, tail])

        name = f"{p.stem}.wav"
        sf.write(out / name, y, sr)

        dur = len(x) / sr
        row = {
            "id": p.stem,
            "file": name,
            "sample_rate": sr,
            # t=0 for every metric in the benchmark.
            "endpoint_sample": int(end),
            "speech_duration_s": round(dur, 3),
            "total_duration_s": round(len(y) / sr, 3),
            "bucket": bucket_for(dur),
            "language": "en",
            "transcript": "",
            "reference_answer": "",
        }
        # Sidecar metadata never overrides anything measured from the audio:
        # the endpoint and the bucket are the harness's own annotations.
        for k, v in refs.get(p.stem, {}).items():
            if k not in {"id", "file", "sample_rate", "endpoint_sample",
                         "speech_duration_s", "total_duration_s", "bucket"}:
                row[k] = v
        rows.append(row)
        print(f"{p.stem}: speech {dur:.2f}s, endpoint at sample {end}")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} clips -> {args.manifest}")


def bucket_for(duration_s: float) -> str:
    """Length buckets.

    Slice results by these. Batch ASR cost scales with utterance length while
    streaming ASR barely moves, so without long clips the streaming win looks
    unimpressive and you draw the wrong conclusion.
    """
    if duration_s < 3:
        return "short"
    if duration_s < 8:
        return "medium"
    return "long"


if __name__ == "__main__":
    main()
