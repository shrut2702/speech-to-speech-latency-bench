"""Moshi: full-duplex speech-to-speech, behind the same interface as the cascade.

Structurally different in a way that matters for the comparison. Moshi listens
and speaks at once, so it has no endpointing step and no notion of "the user has
finished". It decides for itself when to talk, which is why the whole stream is
fed rather than stopping at the endpoint.

That is exactly why t=0 comes from the manifest's annotated end-of-speech rather
than from a VAD. It is a property of the audio file, so both systems are
measured from the identical instant and neither is handed an advantage.

Audio coming back before the endpoint is not suppressed. "Spoke too early" is
real behaviour and the metrics should be able to see it.

Moshi is a single model, so it has none of the stage contention the cascade
suffers when several models share a GPU. Worth stating in the writeup, since it
cuts against the cascade.

Two rate mismatches the harness bridges. The feeder streams 16kHz because that
is what Whisper wants, and Mimi is a 24kHz codec. And Mimi runs at 12.5Hz, one
frame per 80ms, so the feeder's 20ms frames are accumulated before a step. That
80ms is a floor on how finely Moshi can react, and it is a property of the
architecture rather than the host.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from ..feeder import AudioFeeder
from ..trace import Trace, OUTPUT_FIRST_AUDIO, OUTPUT_CHUNK, OUTPUT_END
from .base import AudioChunk, S2SSystem

MIMI_SR = 24000
MIMI_FRAME = 1920  # 80ms at 24kHz


class MoshiSystem(S2SSystem):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg.get("name", "moshi")
        model = cfg.get("model", {})
        self.backend = model.get("backend", "mock")
        self.repo = model.get("repo", "kyutai/moshiko-pytorch-bf16")
        self.device = cfg.get("devices", {}).get("moshi", "cuda")
        self.n_codebooks = int(model.get("codebooks", 8))
        # The feeder pads a long tail of silence because a microphone does not
        # switch off while the user waits. Feeding all of it every trial would
        # cost those seconds for nothing, so the turn ends when the model stops
        # talking. Both bounds are measured from the endpoint.
        self.silence_gap_s = float(model.get("silence_gap_s", 1.5))
        self.max_response_s = float(model.get("max_response_s", 20.0))
        # Mock only: seconds of input consumed before it starts talking. Set it
        # below the clip's endpoint to exercise the "spoke too early" case.
        self.mock_start_s = float(model.get("mock_start_s", 2.0))
        self.mock_response_s = float(model.get("mock_response_s", 4.0))
        self.out_sr = MIMI_SR
        self.mimi = None
        self.lm = None
        self.rtf: float | None = None

    # ---- model ----------------------------------------------------------

    async def load(self) -> None:
        if self.backend == "mock":
            return
        import torch
        from moshi.models import loaders

        ckpt = loaders.CheckpointInfo.from_hf_repo(self.repo)
        self.mimi = ckpt.get_mimi(device=self.device)
        self.mimi.set_num_codebooks(self.n_codebooks)
        self.lm = ckpt.get_moshi(device=self.device)
        self.torch = torch

    async def warmup(self, feeder_factory) -> None:
        """Measures real-time factor before any trial is trusted.

        Above 1.0 the model cannot keep pace with the audio it is fed, every
        number afterwards describes the GPU rather than the architecture, and
        the run belongs on a bigger card.
        """
        if self.backend == "mock":
            return await super().warmup(feeder_factory)

        for _ in range(2):
            await self._run_silence(1.0)
        audio_s = 4.0
        t0 = time.monotonic()
        await self._run_silence(audio_s)
        self.rtf = (time.monotonic() - t0) / audio_s
        if self.rtf >= 1.0:
            raise RuntimeError(
                f"moshi real-time factor is {self.rtf:.2f}, at or above 1.0; "
                "move to a larger GPU before trusting any measurement"
            )

    async def _run_silence(self, seconds: float) -> None:
        state = self._new_state()
        frame = np.zeros(MIMI_FRAME, dtype=np.float32)
        for _ in range(int(seconds * MIMI_SR / MIMI_FRAME)):
            await asyncio.to_thread(self._step, state, frame)

    def _new_state(self):
        from moshi.models import LMGen

        gen = LMGen(self.lm, temp=0.0, temp_text=0.0)
        # Streaming contexts stay open for the session; reopening them per frame
        # would reset the KV cache every 80ms.
        gen.streaming_forever(1)
        self.mimi.streaming_forever(1)
        return gen

    def _step(self, state, frame: np.ndarray):
        with self.torch.no_grad():
            x = self.torch.from_numpy(frame).to(self.device)[None, None, :]
            tokens = state.step(self.mimi.encode(x))
            if tokens is None:
                return None
            # Column 0 is the text stream; the audio codebooks follow.
            return self.mimi.decode(tokens[:, 1:]).cpu().numpy().reshape(-1)

    # ---- the trial ------------------------------------------------------

    async def run(
        self, feeder: AudioFeeder, trace: Trace
    ) -> AsyncIterator[AudioChunk]:
        trace.env.setdefault("moshi_rtf", self.rtf)

        buffer = np.zeros(0, dtype=np.float32)
        state = None if self.backend == "mock" else self._new_state()
        elapsed_s = 0.0
        spoken_s = 0.0
        last_audio: float | None = None
        first_audio = False

        # Past the endpoint, not stopping at it. A full-duplex model never stops
        # listening and can only keep generating while frames keep arriving, so
        # cutting the input early would starve it mid-sentence.
        async for frame in feeder.stream(trace):
            elapsed_s += len(frame.samples) / feeder.sr
            buffer = np.concatenate([buffer, self._to_mimi(frame.samples, feeder.sr)])

            while len(buffer) >= MIMI_FRAME:
                block, buffer = buffer[:MIMI_FRAME], buffer[MIMI_FRAME:]
                if self.backend == "mock":
                    out = self._mock_step(elapsed_s, spoken_s)
                else:
                    out = await asyncio.to_thread(self._step, state, block)
                if out is None or not len(out):
                    continue

                spoken_s += len(out) / MIMI_SR
                last_audio = time.monotonic()
                chunk = AudioChunk(out, self.out_sr)
                if not first_audio:
                    first_audio = True
                    trace.mark(OUTPUT_FIRST_AUDIO)
                trace.mark(OUTPUT_CHUNK, duration_s=chunk.duration_s)
                yield chunk

            if self._turn_over(feeder, last_audio):
                break

        trace.mark(OUTPUT_END)

    def _to_mimi(self, samples: np.ndarray, sr: int) -> np.ndarray:
        if sr == MIMI_SR:
            return samples
        import soxr

        return soxr.resample(samples, sr, MIMI_SR).astype(np.float32)

    def _mock_step(self, elapsed_s: float, spoken_s: float) -> np.ndarray | None:
        if elapsed_s < self.mock_start_s or spoken_s >= self.mock_response_s:
            return None
        return np.zeros(MIMI_FRAME, dtype=np.float32)

    def _turn_over(self, feeder: AudioFeeder, last_audio: float | None) -> bool:
        now = time.monotonic()
        if feeder.t_endpoint is None or now < feeder.t_endpoint:
            return False
        if now - feeder.t_endpoint > self.max_response_s:
            return True
        return last_audio is not None and now - last_audio > self.silence_gap_s
