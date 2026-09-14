"""Cascaded pipeline: ASR -> LLM -> TTS, all in one process.

Three paths, selected by config:

  batch        ASR waits for the whole utterance, the LLM generates its full
               response, TTS synthesizes all of it, then audio plays.
  stream_gen   ASR still batch, but the LLM streams and TTS starts on the first
               chunk while the LLM writes the next.
  stream_all   ASR also runs incrementally during speech, so when the endpoint
               arrives only a short finalize remains.

Stages are plain objects on separate GPUs, not services. A benchmark is a job,
not a production deployment, so putting a network hop between the stages would
only add it to the number being measured.

Chunking is uneven on purpose. The first chunk is cut small to get audio
started; everything after is sentence-sized. Only the first chunk's synthesis
time is ever heard, so later chunks can afford to be long enough to sound
right. See bench/chunking.py.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import AsyncIterator

import numpy as np

from .. import stages
from ..chunking import ChunkPolicy
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
from .base import AudioChunk, S2SSystem, drain_to_endpoint

PATHS = ("batch", "stream_gen", "stream_all")


class CascadeSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.path = cfg["path"]
        if self.path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        self.name = cfg.get("name", f"cascade_{self.path}")
        self.devices = cfg.get("devices", {})
        self.asr = stages.build("asr", cfg["asr"], self.devices)
        self.llm = stages.build("llm", cfg["llm"], self.devices)
        self.tts = stages.build("tts", cfg["tts"], self.devices)
        self.out_sr = int(cfg.get("output_sample_rate", 24000))

    async def load(self) -> None:
        for stage in (self.asr, self.llm, self.tts):
            await stage.load()
        self.out_sr = getattr(self.tts, "sample_rate", self.out_sr)

        # A config claiming one TTS family while another is loaded would put one
        # family's numbers under the other's name and invert the comparison the
        # grid exists for.
        want = self.cfg["tts"].get("family")
        got = getattr(self.tts, "family", None)
        if want and got and want != got:
            raise ValueError(f"config says tts family {want!r}, loaded {got!r}")

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        trace.env.setdefault("devices", self.devices)
        transcript, tail = await self._consume_audio(feeder, trace)

        # The mic keeps running in a real deployment, but a half-duplex cascade
        # has no use for those frames once the transcript is final. Draining the
        # full configured silence would add it to every trial for nothing.
        tail.cancel()
        with suppress(asyncio.CancelledError):
            await tail

        first_audio = False
        async for chunk in self._generate(transcript, trace):
            if not first_audio:
                trace.mark(OUTPUT_FIRST_AUDIO)
                first_audio = True
            trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
            yield chunk

        trace.mark(OUTPUT_END)

    async def _consume_audio(self, feeder: AudioFeeder, trace: Trace):
        """Feeds the ASR and returns (transcript, tail_task).

        Reacts at the endpoint, never at end-of-stream. The file running out is
        an out-of-band signal a live system never receives, and waiting for it
        would add the whole trailing-silence pad to every measurement.
        """
        session = self.asr.new_session()
        saw_partial = False

        async def on_frame(frame) -> None:
            nonlocal saw_partial
            partial = await session.accept(frame.samples)
            if partial and not saw_partial:
                saw_partial = True
                trace.mark(ASR_FIRST_PARTIAL)

        tail = await drain_to_endpoint(feeder, trace, on_frame)

        transcript = await session.final()
        trace.mark(ASR_FINAL, n_chars=len(transcript))
        return transcript, tail

    def _llm_params(self) -> dict:
        llm = self.cfg.get("llm", {})
        return {
            "system": llm.get("system_prompt", ""),
            "max_tokens": int(llm.get("max_tokens", 96)),
            "temperature": float(llm.get("temperature", 0.0)),
            "seed": llm.get("seed", 0),
        }

    def _generate(self, transcript: str, trace: Trace) -> AsyncIterator[AudioChunk]:
        if self.path == "batch":
            return self._generate_batch(transcript, trace)
        return self._generate_streaming(transcript, trace)

    async def _generate_batch(
        self, transcript: str, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        """Nothing overlaps. Full response, then full synthesis, then audio."""
        parts: list[str] = []
        async for token in self.llm.generate(transcript, **self._llm_params()):
            if not parts:
                trace.mark(LLM_FIRST_TOKEN)
            parts.append(token)
        text = "".join(parts)
        trace.mark(LLM_LAST_TOKEN, n_tokens=len(parts), n_chars=len(text))

        pieces = [a async for a in self._synth(text, 0, trace)]
        trace.mark(TTS_LAST_CHUNK)
        yield AudioChunk(
            np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32),
            self.out_sr,
        )

    async def _generate_streaming(
        self, transcript: str, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        """Overlaps synthesis with generation.

        Usually the single largest win available, and it costs no quality. The
        queue keeps the LLM from blocking on TTS.
        """
        tts_cfg = self.cfg.get("tts", {})
        chunks: asyncio.Queue = asyncio.Queue()

        async def produce_text() -> None:
            policy = ChunkPolicy(
                first_chunk_words=int(tts_cfg.get("first_chunk_words", 4)),
                max_words=int(tts_cfg.get("max_chunk_words", 40)),
            )
            n = 0
            try:
                async for token in self.llm.generate(
                    transcript, **self._llm_params()
                ):
                    if not n:
                        trace.mark(LLM_FIRST_TOKEN)
                    n += 1
                    for chunk in policy.feed(token):
                        chunks.put_nowait(chunk)
                for chunk in policy.flush():
                    chunks.put_nowait(chunk)
                trace.mark(LLM_LAST_TOKEN, n_tokens=n)
            finally:
                chunks.put_nowait(None)

        producer = asyncio.create_task(produce_text())
        index = 0
        try:
            while True:
                text = await chunks.get()
                if text is None:
                    break
                async for audio in self._synth(text, index, trace):
                    yield AudioChunk(audio, self.out_sr)
                index += 1
            trace.mark(TTS_LAST_CHUNK)
        finally:
            await producer

    async def _synth(
        self, text: str, index: int, trace: Trace
    ) -> AsyncIterator[np.ndarray]:
        if not text.strip():
            return
        first = True
        async for audio in self.tts.synth(text):
            if first and index == 0:
                trace.mark(TTS_FIRST_CHUNK)
            first = False
            yield audio
