"""Metrics derived from traces.

Everything is measured from the endpoint (`feeder.endpoint`), which is the
annotated end of user speech. Percentiles only: means hide the tail, and the
tail is what a user notices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .trace import (
    Trace,
    FEEDER_ENDPOINT,
    ASR_START,
    ASR_FIRST_PARTIAL,
    ASR_FINAL,
    LLM_START,
    LLM_FIRST_TOKEN,
    LLM_SECOND_TOKEN,
    LLM_LAST_TOKEN,
    TTS_START,
    TTS_FIRST_TOKEN,
    TTS_LAST_TOKEN,
    TTS_FIRST_CHUNK,
    TTS_LAST_CHUNK,
    DECODER_FIRST_CHUNK,
    DECODER_LAST_CHUNK,
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

    # --- from the endpoint: what the user waits through ---------------------
    ttfa_ms: float | None = None            # the headline number
    e2e_ms: float | None = None             # endpoint to the last audio out
    asr_final_ms: float | None = None
    llm_first_token_ms: float | None = None
    llm_last_token_ms: float | None = None
    tts_first_chunk_ms: float | None = None

    # --- from each stage's own start: what that stage costs -----------------
    # "how long did the LLM take" and "how long after the user stopped talking"
    # are different questions. Only the second accumulates the stages before it,
    # so a stage is profiled against its own clock.
    asr_first_partial_ms: float | None = None   # streaming only, from ASR start
    asr_total_ms: float | None = None           # ASR start to final transcript
    llm_ttft_ms: float | None = None            # LLM start to first token
    llm_second_token_ms: float | None = None    # first token to second, one
                                                # decode step with a warm cache
    llm_total_ms: float | None = None           # LLM start to last token
    tts_ttfa_ms: float | None = None            # TTS start to first audio out
    tts_total_ms: float | None = None           # TTS start to last audio out
    # AR families only, and only once the codec-LM is split from its decoder.
    tts_first_token_ms: float | None = None
    tts_tokens_total_ms: float | None = None
    decoder_first_chunk_ms: float | None = None
    decoder_total_ms: float | None = None

    response_audio_s: float | None = None
    rtf: float | None = None                # synthesis wall time / audio produced
    max_gap_ms: float | None = None         # worst inter-chunk gap
    underruns: int | None = None            # gaps that would be audible

    feeder_max_lag_ms: float | None = None
    response_tokens: int | None = None


def _delta_ms(trace: Trace, event, base: float | None) -> float | None:
    t = trace.first(event)
    return None if t is None or base is None else (t - base) * MS


def _span_ms(trace: Trace, start_event, end_event) -> float | None:
    """Duration between two events, or None if either is missing.

    Stages that never ran leave their events unemitted rather than zero, so a
    missing number reads as "not measured" instead of "took no time".
    """
    return _delta_ms(trace, end_event, trace.first(start_event))


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
    m.e2e_ms = _delta_ms(trace, OUTPUT_END, base)
    m.asr_final_ms = _delta_ms(trace, ASR_FINAL, base)
    m.llm_first_token_ms = _delta_ms(trace, LLM_FIRST_TOKEN, base)
    m.llm_last_token_ms = _delta_ms(trace, LLM_LAST_TOKEN, base)
    m.tts_first_chunk_ms = _delta_ms(trace, TTS_FIRST_CHUNK, base)

    m.asr_first_partial_ms = _span_ms(trace, ASR_START, ASR_FIRST_PARTIAL)
    m.asr_total_ms = _span_ms(trace, ASR_START, ASR_FINAL)
    m.llm_ttft_ms = _span_ms(trace, LLM_START, LLM_FIRST_TOKEN)
    m.llm_second_token_ms = _span_ms(trace, LLM_FIRST_TOKEN, LLM_SECOND_TOKEN)
    m.llm_total_ms = _span_ms(trace, LLM_START, LLM_LAST_TOKEN)
    m.tts_ttfa_ms = _span_ms(trace, TTS_START, TTS_FIRST_CHUNK)
    m.tts_total_ms = _span_ms(trace, TTS_START, TTS_LAST_CHUNK)
    m.tts_first_token_ms = _span_ms(trace, TTS_START, TTS_FIRST_TOKEN)
    m.tts_tokens_total_ms = _span_ms(trace, TTS_START, TTS_LAST_TOKEN)
    m.decoder_first_chunk_ms = _span_ms(trace, TTS_FIRST_TOKEN, DECODER_FIRST_CHUNK)
    m.decoder_total_ms = _span_ms(trace, TTS_FIRST_TOKEN, DECODER_LAST_CHUNK)

    if m.ttfa_ms is None:
        m.ok = False
        m.reason = "no audio produced"
        return m

    # The feeder falling behind means the host could not sustain real time, so
    # the trial says more about the machine than the pipeline.
    ends = trace.all_of(("feeder", "end"))
    if ends:
        lag = ends[0].meta.get("max_lag_s", 0.0)
        m.feeder_max_lag_ms = lag * MS
        if lag > 0.05:
            m.ok = False
            m.reason = f"feeder lagged {lag * MS:.0f}ms"

    chunks = trace.all_of(OUTPUT_CHUNK)
    if chunks:
        produced = sum(c.meta.get("duration_s", 0.0) for c in chunks)
        m.response_audio_s = produced or None

        # From when synthesis could begin, not from when the first chunk
        # arrived. Measuring from the first chunk leaves that chunk's audio in
        # the denominator while its synthesis time is not in the numerator, and
        # on the batch path there is only one chunk, so the span collapses to
        # zero and every run reports rtf 0.
        t_end = trace.first(OUTPUT_END) or chunks[-1].t
        t_start = trace.first(TTS_START) or chunks[0].t
        if produced > 0:
            m.rtf = (t_end - t_start) / produced

        # Gaps between consecutive chunks arriving. A gap longer than the audio
        # already buffered is what the listener hears as a stutter.
        gaps = [
            (b.t - a.t) * MS - a.meta.get("duration_s", 0.0) * MS
            for a, b in zip(chunks, chunks[1:])
        ]
        if gaps:
            m.max_gap_ms = max(gaps)
            m.underruns = sum(1 for g in gaps if g > underrun_gap_ms)

    last_token = trace.all_of(LLM_LAST_TOKEN)
    if last_token:
        m.response_tokens = last_token[0].meta.get("n_tokens")

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
