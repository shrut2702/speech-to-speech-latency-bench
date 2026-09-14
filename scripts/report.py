"""Builds the results table from committed traces.

    python scripts/report.py results/*.jsonl

Regenerating the table from traces rather than pasting numbers by hand means a
reader can rerun it, and so can you three weeks later when you have forgotten
which run produced which figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics import TrialMetrics, check_work_constant, summarize, trial_metrics
from bench.trace import load_traces

FIELDS = [
    ("ttfa_ms", "time to first audio"),
    ("asr_final_ms", "asr final"),
    ("llm_first_token_ms", "llm first token"),
    ("tts_first_chunk_ms", "tts first chunk"),
]


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))

    metrics: list[TrialMetrics] = []
    gpus = set()
    for path in args.traces:
        for tr in load_traces(path):
            if tr.warmup:
                continue
            metrics.append(trial_metrics(tr))
            if tr.env.get("gpu"):
                gpus.add(tr.env["gpu"])

    if len(gpus) > 1:
        print(f"!! traces span multiple GPUs {sorted(gpus)}; not comparable\n")

    problems = check_work_constant(metrics)
    if problems:
        print("!! response length varied, latencies are not comparable:")
        for p in problems:
            print(f"   {p}")
        print()

    by_config: dict[str, list[TrialMetrics]] = defaultdict(list)
    for m in metrics:
        by_config[m.config].append(m)

    print("## Overall\n")
    header = "| config | n | " + " | ".join(
        f"{label} p50/p95" for _, label in FIELDS
    ) + " |"
    print(header)
    print("|" + "---|" * (len(FIELDS) + 2))
    for config, ms in sorted(by_config.items()):
        cells = []
        for field, _ in FIELDS:
            s = summarize(ms, field)
            cells.append(f"{s['p50']:.0f} / {s['p95']:.0f}")
        n = summarize(ms, "ttfa_ms")["n"]
        print(f"| {config} | {n} | " + " | ".join(cells) + " |")

    # The averaged number hides the best finding. Streaming ASR saves little on
    # short utterances and a lot on long ones, and only this view shows it.
    if manifest:
        print("\n## Time to first audio by length bucket\n")
        buckets = ["short", "medium", "long"]
        print("| config | " + " | ".join(f"{b} p95" for b in buckets) + " |")
        print("|" + "---|" * (len(buckets) + 1))
        for config, ms in sorted(by_config.items()):
            cells = []
            for b in buckets:
                sel = [
                    m for m in ms if manifest.get(m.clip_id, {}).get("bucket") == b
                ]
                s = summarize(sel, "ttfa_ms")
                cells.append("n/a" if s["n"] == 0 else f"{s['p95']:.0f}")
            print(f"| {config} | " + " | ".join(cells) + " |")

    print("\n## Streaming health\n")
    print("| config | rtf p95 | max gap p95 ms | underruns | failed trials |")
    print("|---|---|---|---|---|")
    for config, ms in sorted(by_config.items()):
        rtf = summarize(ms, "rtf")
        gap = summarize(ms, "max_gap_ms")
        under = sum(m.underruns or 0 for m in ms if m.ok)
        failed = summarize(ms, "ttfa_ms")["n_failed"]
        print(
            f"| {config} | {rtf['p95']:.2f} | {gap['p95']:.0f} | {under} | {failed} |"
        )


if __name__ == "__main__":
    main()
