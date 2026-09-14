"""Metrics derived from traces.

Everything is measured from the endpoint (`feeder.endpoint`), which is the
annotated end of user speech. Percentiles only: means hide the tail, and the
tail is what a user notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

import numpy as np

from .trace import (
    Trace,
    FEEDER_ENDPOINT,
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    LLM_LAST_TOKEN,
    TTS_FIRST_CHUNK,
    OUTPUT_FIRST_AUDIO,
    OUTPUT_CHUNK,
    OUTPUT_END,
)

MS = 1000.0


@dataclass
class TrialMetrics:
    clip_id: str
    config: str
    trial: int
    ok: bool
    reason: str = ""

    ttfa_ms: float | None = None            # the headline number
    asr_final_ms: float | None = None       # all of these are from endpoint
    llm_first_token_ms: float | None = None
    llm_last_token_ms: float | None = None
    tts_first_chunk_ms: float | None = None

    response_audio_s: float | None = None
    rtf: float | None = None                # generation time / audio produced
    max_gap_ms: float | None = None         # worst inter-chunk gap
    underruns: int | None = None            # gaps that would be audible

    feeder_max_lag_ms: float | None = None
    response_tokens: int | None = None


def _delta_ms(trace: Trace, event, base: float) -> float | None:
    t = trace.first(event)
    return None if t is None else (t - base) * MS


def trial_metrics(trace: Trace, underrun_gap_ms: float = 50.0) -> TrialMetrics:
    base = trace.first(FEEDER_ENDPOINT)
    m = TrialMetrics(
        clip_id=trace.clip_id, config=trace.config, trial=trace.trial, ok=True
    )

    if base is None:
        m.ok = False
        m.reason = "no endpoint event"
        return m

    m.ttfa_ms = _delta_ms(trace, OUTPUT_FIRST_AUDIO, base)
    m.asr_final_ms = _delta_ms(trace, ASR_FINAL, base)
    m.llm_first_token_ms = _delta_ms(trace, LLM_FIRST_TOKEN, base)
    m.llm_last_token_ms = _delta_ms(trace, LLM_LAST_TOKEN, base)
    m.tts_first_chunk_ms = _delta_ms(trace, TTS_FIRST_CHUNK, base)

    if m.ttfa_ms is None:
        m.ok = False
        m.reason = "no audio produced"
        return m

    # The feeder falling behind means the host could not sustain real time, so
    # the trial says more about the machine than the pipeline.
    end = trace.first(("feeder", "end"))
    if end is not None:
        lag = next(
            (e.meta.get("max_lag_s", 0.0) for e in trace.all_of(("feeder", "end"))),
            0.0,
        )
        m.feeder_max_lag_ms = lag * MS
        if lag > 0.05:
            m.ok = False
            m.reason = f"feeder lagged {lag * MS:.0f}ms"

    chunks = trace.all_of(OUTPUT_CHUNK)
    if chunks:
        produced = sum(c.meta.get("duration_s", 0.0) for c in chunks)
        m.response_audio_s = produced or None

        t_first = chunks[0].t
        t_end = trace.first(OUTPUT_END) or chunks[-1].t
        gen_time = t_end - t_first
        if produced > 0:
            m.rtf = gen_time / produced

        # Gaps between consecutive chunks arriving. A gap longer than the audio
        # already buffered is what the listener hears as a stutter.
        gaps = [
            (b.t - a.t) * MS - a.meta.get("duration_s", 0.0) * MS
            for a, b in zip(chunks, chunks[1:])
        ]
        if gaps:
            m.max_gap_ms = max(gaps)
            m.underruns = sum(1 for g in gaps if g > underrun_gap_ms)

    tok = trace.first(LLM_LAST_TOKEN)
    if tok is not None:
        m.response_tokens = next(
            (e.meta.get("n_tokens") for e in trace.all_of(LLM_LAST_TOKEN)), None
        )

    return m


def percentiles(values: Iterable[float], ps=(50, 95, 99)) -> dict[str, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {f"p{p}": float("nan") for p in ps}
    arr = np.asarray(vals, dtype=float)
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


def summarize(metrics: list[TrialMetrics], field: str = "ttfa_ms") -> dict:
    """Aggregates one field across trials, excluding warmup and failed trials."""
    good = [m for m in metrics if m.ok]
    vals = [getattr(m, field) for m in good]
    out = percentiles(vals)
    out["n"] = len(good)
    out["n_failed"] = len(metrics) - len(good)
    return out


def check_work_constant(metrics: list[TrialMetrics]) -> list[str]:
    """Validity gate.

    If the LLM produced a different number of tokens across configs for the same
    clip, the configs did different amounts of work and their latencies are not
    comparable. Catch it here rather than at writeup time.
    """
    problems = []
    by_clip: dict[str, set[int]] = {}
    for m in metrics:
        if m.response_tokens is None:
            continue
        by_clip.setdefault(m.clip_id, set()).add(m.response_tokens)
    for clip, counts in by_clip.items():
        if len(counts) > 1:
            problems.append(
                f"{clip}: response length varied across trials/configs {sorted(counts)}"
            )
    return problems
