"""Cascaded pipeline: ASR -> LLM -> TTS -> vocoder.

Three paths, selected by config:

  batch        ASR waits for the whole utterance, the LLM generates its full
               response, TTS synthesizes all of it, then audio plays.
  stream_gen   ASR still batch, but the LLM streams tokens and TTS starts on
               the first sentence while the LLM writes the next.
  stream_all   ASR also runs incrementally during speech, so when the endpoint
               arrives only a short finalize remains.

Note that "streaming" means different things per TTS family. An AR codec-LM TTS
emits acoustic tokens continuously and the codec decodes them incrementally. A
flow-matching model has no tokens to stream, so streaming there means chunking
on sentence boundaries. That asymmetry is a finding, not a flaw: say so in the
writeup rather than presenting a clean grid.

STATUS: skeleton. Model calls are marked TODO.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from ..feeder import AudioFeeder
from ..trace import (
    Trace,
    ASR_FIRST_PARTIAL,
    ASR_FINAL,
    LLM_FIRST_TOKEN,
    LLM_LAST_TOKEN,
    TTS_FIRST_CHUNK,
    TTS_LAST_CHUNK,
    OUTPUT_FIRST_AUDIO,
    OUTPUT_CHUNK,
    OUTPUT_END,
)
from .base import AudioChunk, S2SSystem, drain_to_endpoint, stop_drain

PATHS = ("batch", "stream_gen", "stream_all")


class CascadeSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.path = cfg["path"]
        if self.path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        self.name = cfg.get("name", f"cascade_{self.path}")
        self.asr = None
        self.llm = None
        self.tts = None

    async def load(self) -> None:
        # TODO: faster-whisper for batch, whisper_streaming for stream_all.
        #       Use the SAME Whisper weights for both so the only difference is
        #       the streaming policy, not the model.
        # TODO: LLM served by vLLM. Pin temperature=0 and a system prompt that
        #       caps response length, or TTS does different work per trial and
        #       nothing is comparable.
        # TODO: TTS: CosyVoice2 (AR) or F5-TTS / Kokoro (NAR).
        raise NotImplementedError("wire up ASR, LLM and TTS")

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        transcript, tail = await self._consume_audio(feeder, trace)

        first_audio_emitted = False
        async for chunk in self._generate(transcript, trace):
            if not first_audio_emitted:
                trace.mark(OUTPUT_FIRST_AUDIO)
                first_audio_emitted = True
            trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
            yield chunk

        # Half-duplex: the remaining silence is of no use to us.
        await stop_drain(tail)
        trace.mark(OUTPUT_END)

    async def _consume_audio(self, feeder: AudioFeeder, trace: Trace):
        """Consumes audio up to the endpoint and returns (transcript, tail_task).

        In `stream_all` the ASR runs as frames arrive, so most of the work has
        already happened by the time the endpoint fires and only a finalize
        remains. In the other paths frames are merely buffered and the whole
        utterance is transcribed afterwards, which is why utterance length hurts
        those paths and barely touches this one.

        Note that we react at the endpoint rather than waiting for the stream to
        end. Waiting would add the entire trailing-silence pad to every number.
        """
        buffer: list[np.ndarray] = []
        saw_first_partial = False

        async def on_frame(frame) -> None:
            nonlocal saw_first_partial
            buffer.append(frame.samples)

            if self.path == "stream_all" and not frame.past_endpoint:
                # TODO: feed the frame to streaming ASR, get a partial hypothesis
                if not saw_first_partial:
                    trace.mark(ASR_FIRST_PARTIAL)
                    saw_first_partial = True

        tail = await drain_to_endpoint(feeder, trace, on_frame)

        audio = np.concatenate(buffer) if buffer else np.zeros(0, dtype=np.float32)

        # TODO: batch transcribe (batch / stream_gen) or finalize (stream_all)
        transcript = ""
        trace.mark(ASR_FINAL, n_chars=len(transcript))
        return transcript, tail

    async def _generate(
        self, transcript: str, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        if self.path == "batch":
            async for c in self._generate_batch(transcript, trace):
                yield c
        else:
            async for c in self._generate_streaming(transcript, trace):
                yield c

    async def _generate_batch(
        self, transcript: str, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        # TODO: full LLM generation, then synthesize the entire response.
        trace.mark(LLM_FIRST_TOKEN)
        text = ""
        trace.mark(LLM_LAST_TOKEN, n_tokens=0)

        trace.mark(TTS_FIRST_CHUNK)
        audio = np.zeros(0, dtype=np.float32)
        trace.mark(TTS_LAST_CHUNK)
        yield AudioChunk(audio, self.cfg.get("output_sample_rate", 24000))

    async def _generate_streaming(
        self, transcript: str, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        """Overlaps synthesis with generation.

        Usually the single largest win available, and it costs no quality. A
        queue between the two stages keeps the LLM from blocking on TTS.
        """
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        async def produce_text() -> None:
            # TODO: stream tokens from vLLM, cut on sentence boundaries, push.
            trace.mark(LLM_FIRST_TOKEN)
            trace.mark(LLM_LAST_TOKEN, n_tokens=0)
            await sentences.put(None)

        producer = asyncio.create_task(produce_text())
        first_tts_chunk = False
        try:
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                # TODO: AR TTS -> stream acoustic tokens, decode incrementally.
                #       NAR TTS -> synthesize this sentence in one shot.
                if not first_tts_chunk:
                    trace.mark(TTS_FIRST_CHUNK)
                    first_tts_chunk = True
                yield AudioChunk(
                    np.zeros(0, dtype=np.float32),
                    self.cfg.get("output_sample_rate", 24000),
                )
            trace.mark(TTS_LAST_CHUNK)
        finally:
            await producer
