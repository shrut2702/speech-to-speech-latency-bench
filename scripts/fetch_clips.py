"""Pulls the benchmark clip set from HuggingFace and stages it for prepare_clips.

Three sources, chosen so the set spans short factual queries through multi-step
reasoning prompts:

  llama-questions   short general-knowledge questions, one-word answers
  MLCpro-en         spoken arithmetic, answers stated as a sentence
  Gsm8kEval         multi-step word problems, long chain-of-thought answers

Every row carries the input text and a reference answer. Both are logged: the
transcript scores ASR, and the reference answer is the only way to tell whether
a latency win cost accuracy. Error propagation, where an ASR mistake becomes a
wrong answer, is the cascade's main structural weakness, and it is invisible
without the answer side.

Selection is stratified by trimmed speech duration using the same annotation
function prepare_clips.py applies later, so the buckets chosen here are the
buckets that land in the manifest. Length matters: batch ASR decode cost grows
with utterance length while streaming ASR barely moves.

    python scripts/fetch_clips.py --out raw --per-source 15

Writes raw wavs plus refs.json, an id to metadata sidecar consumed by
prepare_clips.py --refs. Audio is not committed; this script regenerates it.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

BUCKETS = ("short", "medium", "long")
REVISION = "refs/convert/parquet"

# Each source contributes 15 clips, but no source spans all three length
# buckets: llama-questions tops out under 5s, Gsm8kEval never goes under 4s,
# and MLCpro-en holds only four clips past 8s. So the quotas are skewed per
# source to make the *combined* set come out at 15 per bucket. Any other split
# either drops a bucket or stops taking 15 from each.
SOURCES = [
    {
        "key": "llamaq",
        "repo": "fixie-ai/llama-questions",
        "files": ["default/test/0000.parquet"],
        "audio": "audio",
        "transcript": "question",
        "answer": "answer",
        "extra": {},
        "quota": {"short": 10, "medium": 5, "long": 0},
    },
    {
        "key": "mlcpro",
        "repo": "Honggao/URO-Bench",
        "files": ["MLCpro-en/test/0000.parquet"],
        "audio": "source_wav",
        "transcript": "source_text",
        "answer": "target_text",
        "extra": {},
        "quota": {"short": 5, "medium": 6, "long": 4},
    },
    {
        "key": "gsm8k",
        "repo": "Honggao/URO-Bench",
        "files": ["Gsm8kEval/test/0000.parquet"],
        "audio": "source_wav",
        "transcript": "source_text",
        "answer": "target_text",
        # The short final answer, far easier to score than the full reasoning.
        "extra": {"reference_short": "reference"},
        "quota": {"short": 0, "medium": 4, "long": 11},
    },
]


def _load_prepare_clips():
    """Imports the sibling script so the endpoint annotation cannot drift."""
    path = Path(__file__).with_name("prepare_clips.py")
    spec = importlib.util.spec_from_file_location("prepare_clips", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode(cell):
    """Parquet Audio cells carry raw file bytes."""
    x, sr = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr


def silence_floor(x, sr):
    """Noise floor of the first 200ms, plus the fraction of exact zeros.

    A crude provenance signal. Human recordings carry a noise floor; synthetic
    speech is often digitally silent before the first word. Reported rather than
    acted on, because ASR finds synthetic speech unrealistically easy, and that
    matters for any quality claim drawn from this set.
    """
    head = x[: int(0.2 * sr)]
    if not len(head):
        return 0.0, 0.0
    rms = float(np.sqrt((head.astype(np.float64) ** 2).mean() + 1e-20))
    return 20 * np.log10(rms + 1e-20), float((head == 0).mean())


def pick(rows, k):
    """The k clips closest to the bucket's median length.

    Deterministic, and keeps a bucket from being represented by its own
    extremes. Ties break on clip id, so a rerun gives the identical set.
    """
    if k <= 0 or not rows:
        return []
    mid = sorted(rows, key=lambda r: r[3])[len(rows) // 2][3]
    return sorted(rows, key=lambda r: (abs(r[3] - mid), r[0]))[:k]


def scale_quota(quota, total):
    """Scales a per-source bucket quota to `total` clips, keeping its shape."""
    base = sum(quota.values())
    if total == base:
        return dict(quota)
    want = {b: int(round(q * total / base)) for b, q in quota.items()}
    drift = total - sum(want.values())
    if drift:
        want[max(want, key=lambda b: want[b])] += drift
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="raw")
    ap.add_argument("--per-source", type=int, default=15)
    ap.add_argument("--max-speech-s", type=float, default=20.0,
                    help="the feeder streams at 1x, so clip length is wall-clock cost")
    ap.add_argument("--min-speech-s", type=float, default=0.8)
    ap.add_argument("--survey-only", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    pc = _load_prepare_clips()

    out = Path(args.out)
    refs = {}
    picked_total = 0

    for src in SOURCES:
        cands = {b: [] for b in BUCKETS}
        rates = set()
        floors = []

        for fname in src["files"]:
            print("downloading %s: %s %s" % (src["key"], src["repo"], fname))
            local = hf_hub_download(
                repo_id=src["repo"], filename=fname,
                revision=REVISION, repo_type="dataset",
            )
            table = pq.read_table(local)
            wanted = {src["audio"], src["transcript"], src["answer"]}
            wanted.update(src["extra"].values())
            cols = {c: table.column(c).to_pylist() for c in wanted}
            n = len(cols[src["audio"]])
            print("  %d rows" % n)

            for i in range(n):
                x, sr = decode(cols[src["audio"]][i])
                rates.add(sr)
                end = pc.last_speech_sample(x, sr)
                dur = end / sr
                if not (args.min_speech_s <= dur <= args.max_speech_s):
                    continue
                dbfs, zfrac = silence_floor(x, sr)
                floors.append(dbfs)
                meta = {
                    "transcript": str(cols[src["transcript"]][i]),
                    "reference_answer": str(cols[src["answer"]][i]),
                    "source": src["key"],
                }
                for k, col in src["extra"].items():
                    meta[k] = str(cols[col][i])
                cid = "%s_%04d" % (src["key"], i)
                cands[pc.bucket_for(dur)].append((cid, x[:end], sr, dur, meta, zfrac))

        print("  sample rates: %s" % sorted(rates))
        if floors:
            print("  pre-speech floor: median %.1f dBFS" % float(np.median(floors)))
        for b in BUCKETS:
            rows = cands[b]
            if rows:
                d = sorted(r[3] for r in rows)
                print("  %-7s n=%4d  min %5.2fs  p50 %5.2fs  max %5.2fs"
                      % (b, len(rows), d[0], d[len(d) // 2], d[-1]))
            else:
                print("  %-7s n=0" % b)

        if args.survey_only:
            print()
            continue

        out.mkdir(parents=True, exist_ok=True)

        want = scale_quota(src["quota"], args.per_source)
        taken = {}
        for b in BUCKETS:
            taken[b] = pick(cands[b], want[b])
            if len(taken[b]) < want[b]:
                print("  SHORTFALL: %s/%s has %d, wanted %d"
                      % (src["key"], b, len(cands[b]), want[b]))

        # Keep the per-source count whole even when a bucket ran dry, by topping
        # up from whichever bucket still has unused candidates.
        deficit = args.per_source - sum(len(v) for v in taken.values())
        while deficit > 0:
            spare = sorted(
                BUCKETS,
                key=lambda b: len(cands[b]) - len(taken[b]),
                reverse=True,
            )[0]
            if len(cands[spare]) - len(taken[spare]) <= 0:
                print("  cannot reach %d for %s" % (args.per_source, src["key"]))
                break
            taken[spare] = pick(cands[spare], len(taken[spare]) + 1)
            deficit -= 1

        for b in BUCKETS:
            for cid, x, sr, dur, meta, zfrac in sorted(taken[b], key=lambda r: r[0]):
                sf.write(out / (cid + ".wav"), x, sr)
                meta = dict(meta)
                meta["bucket_at_fetch"] = b
                meta["speech_s"] = round(dur, 3)
                refs[cid] = meta
                picked_total += 1
                print("  %s  %5.2fs  zeros=%3.0f%%  %.44s -> %.28s"
                      % (cid, dur, zfrac * 100,
                         meta["transcript"], meta["reference_answer"]))
        print()

    if args.survey_only:
        return

    (out / "refs.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote %d clips -> %s/ (plus refs.json)" % (picked_total, out))
    print("next: python scripts/prepare_clips.py --src raw --out data/clips "
          "--manifest data/manifest.jsonl --refs raw/refs.json")


if __name__ == "__main__":
    main()
