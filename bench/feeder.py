"""Wall-clock audio feeder.

This is the single most important file in the repo.

A conversational system cannot see the future: audio arrives at 1x speed and it
must decide what to do with each frame as it lands. Handing a model an entire
utterance at once and timing the call measures throughput, not latency, and it
makes streaming ASR look identical to batch ASR because both get all the audio
instantly.

So the feeder emits fixed-size frames against a real clock, exactly like a
microphone would. Every system under test consumes the same stream.

t = 0 for all latency metrics is the annotated end-of-speech sample from the
manifest, converted to a scheduled wall-clock instant. It is a property of the
audio file rather than of any system, so it is identical across configs and
deterministic across runs. No VAD is involved: injecting a perfect oracle
endpoint keeps the comparison about pipelines rather than about a threshold.

The stream does not stop when the clip does. After the file is exhausted the
feeder keeps emitting silence for `silence_after_s`, because a microphone does
not switch off while the user waits for an answer. This matters most for
full-duplex models: Moshi consumes an input stream continuously and can only
keep generating its response while frames keep arriving, so cutting the input
at end-of-file would starve it mid-sentence. It also means the trailing pad
baked into the clip files is cosmetic, and the real mechanism lives here.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

import numpy as np

from .trace import Trace, FEEDER_START, FEEDER_ENDPOINT, FEEDER_END


@dataclass
class Frame:
    index: int
    samples: np.ndarray
    t_emit: float            # when this frame was actually handed over
    t_scheduled: float       # when it should have been handed over
    past_endpoint: bool      # speech has ended; this is trailing silence
    synthetic: bool = False  # generated after the clip ran out, not from file


class AudioFeeder:
    """Streams `samples` as `frame_ms` frames paced on `time.monotonic()`."""

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        endpoint_sample: int,
        frame_ms: int = 20,
        silence_after_s: float = 0.0,
    ):
        if samples.ndim != 1:
            raise ValueError("expected mono audio")
        if not 0 <= endpoint_sample <= len(samples):
            raise ValueError("endpoint_sample outside clip")
        if silence_after_s < 0:
            raise ValueError("silence_after_s must be >= 0")

        self.samples = samples.astype(np.float32, copy=False)
        self.sr = sample_rate
        self.frame_len = int(round(sample_rate * frame_ms / 1000.0))
        self.endpoint_sample = endpoint_sample
        self.silence_after_s = silence_after_s

        self.t_start: float | None = None
        self.t_endpoint: float | None = None
        # How far behind schedule we ever fell. If this is not ~0 the host was
        # too loaded to feed in real time and the trial should be discarded.
        self.max_lag: float = 0.0

    @property
    def n_clip_frames(self) -> int:
        """Frames backed by actual audio from the file."""
        return max(1, math.ceil(len(self.samples) / self.frame_len))

    @property
    def n_silence_frames(self) -> int:
        """Frames of silence emitted after the clip runs out."""
        return int(round(self.silence_after_s * self.sr / self.frame_len))

    @property
    def n_frames(self) -> int:
        return self.n_clip_frames + self.n_silence_frames

    async def stream(self, trace: Trace | None = None):
        """Yields `Frame` objects at wall-clock pace."""
        self.t_start = time.monotonic()
        # Deterministic: derived from the manifest, not observed at runtime.
        self.t_endpoint = self.t_start + self.endpoint_sample / self.sr
        self.max_lag = 0.0

        if trace is not None:
            trace.mark(FEEDER_START, sample_rate=self.sr, frame_len=self.frame_len)

        endpoint_marked = False
        n_clip = self.n_clip_frames
        silence = np.zeros(self.frame_len, dtype=np.float32)

        # The consumer ends the stream by breaking out of its `async for`, which
        # closes this generator. Systems should do that once their response is
        # complete rather than sitting through the whole silence budget.
        for i in range(self.n_frames):
            scheduled = self.t_start + (i * self.frame_len) / self.sr

            delay = scheduled - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                self.max_lag = max(self.max_lag, -delay)

            start = i * self.frame_len
            is_synthetic = i >= n_clip

            if is_synthetic:
                chunk = silence
            else:
                chunk = self.samples[start : start + self.frame_len]
                if len(chunk) < self.frame_len:
                    chunk = np.pad(chunk, (0, self.frame_len - len(chunk)))

            past = start >= self.endpoint_sample
            if past and not endpoint_marked:
                endpoint_marked = True
                if trace is not None:
                    # The scheduled instant, not the moment the loop noticed it.
                    trace.mark_at(
                        FEEDER_ENDPOINT,
                        self.t_endpoint,
                        endpoint_sample=self.endpoint_sample,
                    )

            yield Frame(
                index=i,
                samples=chunk,
                t_emit=time.monotonic(),
                t_scheduled=scheduled,
                past_endpoint=past,
                synthetic=is_synthetic,
            )

        if trace is not None:
            trace.mark(
                FEEDER_END,
                max_lag_s=self.max_lag,
                exhausted=True,
            )
