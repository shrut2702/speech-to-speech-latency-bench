"""Moshi: full-duplex speech-to-speech.

Structurally different from the cascade in a way that matters for the
comparison. Moshi listens and speaks at once, so it has no endpointing step and
no notion of "the user has finished". It decides for itself when to talk.

That is exactly why t=0 comes from the manifest's annotated end-of-speech rather
than from a VAD. It is a property of the audio file, so both systems are
measured from the identical instant and neither is handed an advantage.

Moshi is also a single model, so it has none of the internal stage contention
the cascade suffers when several models share one GPU. Worth stating in the
writeup, since it cuts against the cascade.

STATUS: skeleton. Model calls are marked TODO.
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np

from ..feeder import AudioFeeder
from ..trace import Trace, OUTPUT_FIRST_AUDIO, OUTPUT_CHUNK, OUTPUT_END
from .base import AudioChunk, S2SSystem


class MoshiSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "moshi")
        self.model = None

    async def load(self) -> None:
        # TODO: load Moshi (kyutai-labs/moshi) and the Mimi codec.
        # Check real-time factor on this GPU before trusting any number: if RTF
        # exceeds 1 the results describe the hardware, not the architecture.
        raise NotImplementedError("wire up Moshi")

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        first_audio_emitted = False

        async for frame in feeder.stream(trace):
            # TODO: Mimi-encode the frame, step the model, Mimi-decode output.
            # Moshi runs continuously, so it may emit audio before the endpoint.
            # Do not suppress that: "spoke too early" is a real behaviour and
            # the metrics should be able to see it.
            out: np.ndarray | None = None

            if out is not None and len(out):
                if not first_audio_emitted:
                    trace.mark(OUTPUT_FIRST_AUDIO)
                    first_audio_emitted = True
                chunk = AudioChunk(out, self.cfg.get("output_sample_rate", 24000))
                trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
                yield chunk

        # TODO: keep stepping the model past the input so the response finishes,
        # bounded by a max response duration.

        trace.mark(OUTPUT_END)
