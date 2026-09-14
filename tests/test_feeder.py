"""The feeder is the methodology, so it gets the tests."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from bench.feeder import AudioFeeder
from bench.trace import Trace, FEEDER_ENDPOINT


SR = 16000


def make_feeder(seconds=1.0, endpoint_s=0.6, **kw) -> AudioFeeder:
    samples = np.zeros(int(SR * seconds), dtype=np.float32)
    return AudioFeeder(
        samples=samples,
        sample_rate=SR,
        endpoint_sample=int(SR * endpoint_s),
        **kw,
    )


def test_streams_in_real_time():
    """A 1s clip must take ~1s to feed. If it returns instantly the whole
    benchmark is measuring throughput instead of latency."""

    async def run():
        feeder = make_feeder(seconds=1.0)
        t0 = time.monotonic()
        async for _ in feeder.stream():
            pass
        return time.monotonic() - t0

    elapsed = asyncio.run(run())
    assert 0.95 < elapsed < 1.25, f"expected ~1s, got {elapsed:.3f}s"


def test_does_not_drift():
    """Frame deadlines are absolute, so error must not accumulate."""

    async def run():
        feeder = make_feeder(seconds=2.0)
        worst = 0.0
        async for frame in feeder.stream():
            worst = max(worst, abs(frame.t_emit - frame.t_scheduled))
        return worst

    worst = asyncio.run(run())
    assert worst < 0.05, f"drifted {worst * 1000:.0f}ms"


def test_endpoint_is_deterministic():
    """t=0 comes from the manifest, so it must land at the scheduled instant
    regardless of when the consumer loop noticed it."""

    async def run():
        feeder = make_feeder(seconds=1.0, endpoint_s=0.6)
        trace = Trace(clip_id="t", config="t", trial=0)
        async for _ in feeder.stream(trace):
            await asyncio.sleep(0)  # a slow consumer must not move the endpoint
        offset = trace.first(FEEDER_ENDPOINT) - feeder.t_start
        return offset

    offset = asyncio.run(run())
    assert abs(offset - 0.6) < 1e-6, f"endpoint at {offset:.4f}s, expected 0.6s"


def test_frames_are_uniform():
    async def run():
        feeder = make_feeder(seconds=0.5, endpoint_s=0.3, frame_ms=20)
        lengths = [len(f.samples) async for f in feeder.stream()]
        return lengths

    lengths = asyncio.run(run())
    assert set(lengths) == {int(SR * 0.02)}, "last frame should be zero-padded"


def test_emits_silence_past_the_clip():
    """The mic does not switch off when the file runs out.

    Full-duplex models keep generating only while frames keep arriving, so
    cutting the input at end-of-file would starve them mid-response.
    """

    async def run():
        feeder = make_feeder(seconds=0.4, endpoint_s=0.2, silence_after_s=0.4)
        real, synth = 0, 0
        async for frame in feeder.stream():
            if frame.synthetic:
                synth += 1
                assert not frame.samples.any(), "synthetic frames must be silent"
            else:
                real += 1
        return real, synth

    real, synth = asyncio.run(run())
    assert real == 20, real      # 0.4s / 20ms
    assert synth == 20, synth    # 0.4s / 20ms


def test_consumer_can_end_the_stream_early():
    """Breaking out must stop the feeder, so a finished response does not sit
    through the whole silence budget."""

    async def run():
        feeder = make_feeder(seconds=0.2, endpoint_s=0.1, silence_after_s=30.0)
        t0 = time.monotonic()
        seen = 0
        async for _frame in feeder.stream():
            seen += 1
            if seen == 15:
                break
        return time.monotonic() - t0

    elapsed = asyncio.run(run())
    assert elapsed < 1.0, f"took {elapsed:.2f}s; generator did not close"


def test_rejects_bad_endpoint():
    with pytest.raises(ValueError):
        AudioFeeder(np.zeros(100, dtype=np.float32), SR, endpoint_sample=999)
