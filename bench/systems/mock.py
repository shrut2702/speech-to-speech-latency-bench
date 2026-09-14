"""A fake system with configurable delays.

Exists so the harness can be validated without a GPU. Because you dial in the
stage delays yourself, you know what the report *should* say, which is the only
way to be sure metrics.py is computing what you think it is.

Also useful as a regression test: if a refactor changes the mock's numbers, the
measurement layer moved.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import numpy as np

from ..feeder import AudioFeeder
from ..trace import (
    Trace,
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


class MockSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "mock")
        d = cfg.get("delays_ms", {})
        self.asr_ms = d.get("asr_final", 300)
        self.llm_ttft_ms = d.get("llm_first_token", 150)
        self.tts_ms = d.get("tts_first_chunk", 120)
        self.chunk_ms = d.get("chunk_interval", 80)
        self.n_chunks = cfg.get("n_chunks", 12)
        self.sr = cfg.get("output_sample_rate", 24000)
        self.n_tokens = cfg.get("n_tokens", 42)

    async def load(self) -> None:
        return None

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        # React at the endpoint, keep consuming the tail in the background.
        tail = await drain_to_endpoint(feeder, trace)

        await asyncio.sleep(self.asr_ms / 1000)
        trace.mark(ASR_FINAL)

        await asyncio.sleep(self.llm_ttft_ms / 1000)
        trace.mark(LLM_FIRST_TOKEN)

        await asyncio.sleep(self.tts_ms / 1000)
        trace.mark(TTS_FIRST_CHUNK)

        chunk_s = self.chunk_ms / 1000
        samples = np.zeros(int(self.sr * chunk_s), dtype=np.float32)
        for i in range(self.n_chunks):
            if i:
                await asyncio.sleep(chunk_s)
            if i == 0:
                trace.mark(OUTPUT_FIRST_AUDIO)
            chunk = AudioChunk(samples, self.sr)
            trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
            yield chunk

        trace.mark(LLM_LAST_TOKEN, n_tokens=self.n_tokens)
        trace.mark(TTS_LAST_CHUNK)
        await stop_drain(tail)
        trace.mark(OUTPUT_END)
