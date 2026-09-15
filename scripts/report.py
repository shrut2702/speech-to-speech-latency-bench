"""Builds the results table from committed traces.

    python scripts/report.py results/*.jsonl

Regenerating the table from traces rather than pasting numbers by hand means a
reader can rerun it, and so can you three weeks later when you have forgotten
which run produced which figure.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.metrics import TrialMetrics, check_work_constant, summarize, trial_metrics
from bench.runner import load_manifest
from bench.trace import load_traces

# From the endpoint: what the user actually waits through.
PIPELINE = [
    ("ttfa_ms", "time to first audio"),
    ("e2e_ms", "end to end"),
    ("asr_final_ms", "asr final"),
    ("llm_first_token_ms", "llm first token"),
    ("tts_first_chunk_ms", "tts first chunk"),
]

# From each stage's own start, with the stages before it subtracted out.
STAGES = [
    ("asr_first_partial_ms", "asr first partial"),
    ("asr_total_ms", "asr total"),
    ("llm_ttft_ms", "llm ttft"),
    ("llm_second_token_ms", "llm 2nd token"),
    ("llm_total_ms", "llm total"),
    ("tts_ttfa_ms", "tts first audio"),
    ("tts_total_ms", "tts total"),
]

# AR families only, and unemitted until the codec-LM is split from its decoder.
AR_STAGES = [
    ("tts_first_token_ms", "tts first token"),
    ("tts_tokens_total_ms", "tts tokens total"),
    ("decoder_first_chunk_ms", "decoder first chunk"),
    ("decoder_total_ms", "decoder total"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    args = ap.parse_args()

    manifest = {r["id"]: r for r in load_manifest(args.manifest)}

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

    def table(title: str, fields, note: str = "") -> None:
        """One row per config, p50/p95 per field. Dashes where unmeasured.

        A stage that never ran leaves its events unemitted, so the cell reads
        as "not measured" rather than as zero milliseconds.
        """
        rows = []
        for config, ms in sorted(by_config.items()):
            cells = []
            for field, _ in fields:
                st = summarize(ms, field)
                missing = st["n"] == 0 or st["p50"] != st["p50"]
                cells.append("-" if missing else f"{st['p50']:.0f} / {st['p95']:.0f}")
            if any(c != "-" for c in cells):
                rows.append((config, summarize(ms, "ttfa_ms")["n"], cells))
        if not rows:
            return
        print(f"\n## {title}\n")
        if note:
            print(f"{note}\n")
        print("| config | n | " + " | ".join(f"{l} p50/p95" for _, l in fields) + " |")
        print("|" + "---|" * (len(fields) + 2))
        for config, n, cells in rows:
            print(f"| {config} | {n} | " + " | ".join(cells) + " |")

    table("Pipeline, from end of speech", PIPELINE)
    table("Per stage, from each stage's own start", STAGES,
          "Stage costs with everything before them subtracted out. This is the "
          "view that says which millisecond to go delete.")
    table("AR TTS internals", AR_STAGES,
          "Codec-LM and decoder split. Empty until the split is wired.")

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
