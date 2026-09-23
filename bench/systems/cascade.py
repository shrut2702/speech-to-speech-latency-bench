"""Cascaded pipeline: ASR -> LLM -> TTS, all in one process.

Three paths, selected by config:

  batch        ASR waits for the whole utterance, the LLM generates its full
               response, TTS synthesizes all of it, then audio plays.
  stream_gen   ASR still batch, but the LLM streams and TTS starts on the first
               chunk while the LLM writes the next.
  stream_all   ASR also runs incrementally during speech, so when the endpoint
               arrives only a short finalize remains.

Each stage runs in its own process on its own GPU. Not for deployment realism:
Python threads share one interpreter lock, so in a single process the LLM and
the TTS would take turns rather than overlap, and the streaming paths exist
precisely to overlap them. Profiling CosyVoice2 measured that cost directly.
Every config uses the same three processes, including `batch` where nothing
overlaps, so the paths differ only in what they overlap.

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

from ..chunking import ChunkPolicy
from ..feeder import AudioFeeder
from ..trace import (
    Trace,
    ASR_START,
    ASR_FIRST_PARTIAL,
    ASR_FINAL,
    LLM_START,
    LLM_FIRST_TOKEN,
    LLM_SECOND_TOKEN,
    LLM_LAST_TOKEN,
    TTS_START,
    TTS_FIRST_CHUNK,
    TTS_LAST_CHUNK,
    OUTPUT_FIRST_AUDIO,
    OUTPUT_CHUNK,
    OUTPUT_END,
)
from ..workers import StageWorker, start_all
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
        self.asr = StageWorker("asr", cfg["asr"], self.devices.get("asr"))
        self.llm = StageWorker("llm", cfg["llm"], self.devices.get("llm"))
        self.tts = StageWorker("tts", cfg["tts"], self.devices.get("tts"))
        self.out_sr = int(cfg.get("output_sample_rate", 24000))
        # Batch synthesizes the whole response in one call. Streaming asks the
        # TTS to emit as it goes, which only an AR family can actually do.
        self.tts_stream = self.path != "batch"

    async def load(self) -> None:
        await start_all([self.asr, self.llm, self.tts])

    async def unload(self) -> None:
        for worker in (self.asr, self.llm, self.tts):
            await worker.stop()

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
        await self.asr.call("asr_open")
        started = False
        # When transcription actually begins, which differs by path. Streaming
        # starts on the first frame because its whole point is the work done
        # before t=0. Batch only buffers during speech, so starting its clock at
        # the first frame would report the user's speaking time as ASR cost.
        incremental = self.path == "stream_all"

        async def on_frame(frame) -> None:
            nonlocal started
            if incremental and not started:
                started = True
                trace.mark(ASR_START)
            # Frames are fire and forget. At 50 a second, a round trip each
            # would cost more than the work does.
            self.asr.send("asr_frame", frame.samples)

        tail = await drain_to_endpoint(feeder, trace, on_frame)

        if not incremental:
            trace.mark(ASR_START)
        transcript = await self.asr.call("asr_final")
        trace.mark(ASR_FINAL, n_chars=len(transcript))

        # Partials arrive out of band, already stamped by the ASR process. Only
        # the incremental path produces one that means anything: a batch backend
        # offering a partial is not doing the work early.
        partials = [t for kind, t in self.asr.drain_events() if kind == "partial"]
        if incremental and partials:
            trace.mark_at(ASR_FIRST_PARTIAL, min(partials))
        trace.artifacts["transcript"] = transcript
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
        trace.mark(LLM_START)
        async for token in self.llm.stream("llm", {"prompt": transcript, **self._llm_params()}):
            if not parts:
                trace.mark(LLM_FIRST_TOKEN)
            elif len(parts) == 1:
                trace.mark(LLM_SECOND_TOKEN)
            parts.append(token)
        text = "".join(parts)
        trace.mark(LLM_LAST_TOKEN, n_tokens=len(parts), n_chars=len(text))
        trace.artifacts["response"] = text
        # Batch synthesizes the whole response in one call, so there is one
        # chunk and it is the response.
        trace.artifacts["tts_chunks"] = [text]

        trace.mark(TTS_START)
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
        # Kept so the report can show exactly where the chunker cut. Chunk one
        # sets time to first audio, so its text is the thing to look at when the
        # number moves.
        sent: list[str] = []
        cut: list[str] = []

        async def produce_text() -> None:
            policy = ChunkPolicy(
                first_chunk_words=int(tts_cfg.get("first_chunk_words", 8)),
                min_words=int(tts_cfg.get("min_chunk_words", 6)),
                max_words=int(tts_cfg.get("max_chunk_words", 40)),
            )
            n = 0
            trace.mark(LLM_START)
            try:
                async for token in self.llm.stream(
                    "llm", {"prompt": transcript, **self._llm_params()}
                ):
                    if not n:
                        trace.mark(LLM_FIRST_TOKEN)
                    elif n == 1:
                        trace.mark(LLM_SECOND_TOKEN)
                    n += 1
                    sent.append(token)
                    for chunk in policy.feed(token):
                        cut.append(chunk)
                        chunks.put_nowait(chunk)
                for chunk in policy.flush():
                    cut.append(chunk)
                    chunks.put_nowait(chunk)
                trace.mark(LLM_LAST_TOKEN, n_tokens=n)
            finally:
                trace.artifacts["response"] = "".join(sent)
                trace.artifacts["tts_chunks"] = list(cut)
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
        if index == 0:
            # TTS_START is the first chunk handed over, so tts latency is
            # measured from when synthesis could begin rather than from t=0.
            trace.mark(TTS_START)
        first = True
        async for audio in self.tts.stream(
            "tts", {"text": text, "stream": self.tts_stream}
        ):
            if first and index == 0:
                trace.mark(TTS_FIRST_CHUNK)
            first = False
            yield audio
