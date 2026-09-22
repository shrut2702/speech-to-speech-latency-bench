"""The pipeline stages, as plain objects in one process.

Everything runs inside a single container with several GPUs attached, so stages
talk by calling each other rather than over a socket. Each one takes a `device`
from the config and stays on it.

vLLM is the exception: it claims `cuda:0` and has no clean per-instance device
argument, so give it device 0 and place the others above it.

No service layer, deliberately. Splitting the stages into separate processes
would buy dependency isolation and a production-shaped deployment, and cost a
network hop inside the number being measured. A benchmark is a job, not a
service.

Each stage exposes the smallest surface the cascade needs:

  ASR   new_session() -> accept(frame) during speech, final() at the endpoint
  LLM   generate() -> async iterator of token text
  TTS   synth() -> async iterator of audio chunks

The mock implementations exist so the harness can be validated without a GPU.
You dial in the delay, so you know what the report should say, which is the only
way to be sure the measurement layer computes what you think it does.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator, Protocol

import numpy as np


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------

class ASRSession(Protocol):
    async def accept(self, samples: np.ndarray) -> str | None:
        """Consumes one frame. Returns a partial hypothesis if one is ready.

        Batch backends buffer here and do nothing else. Streaming backends run
        incremental inference, which moves their compute to before t=0 where it
        stops counting against the latency.
        """

    async def final(self) -> str:
        """The finalized transcript. Called at the endpoint, never at EOF."""


class FasterWhisperASR:
    """Batch ASR on faster-whisper. Used by the batch and stream_gen paths."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.model_name = cfg.get("model", "large-v3-turbo")
        self.compute_type = cfg.get("compute_type", "float16")
        self.language = cfg.get("language", "en")
        self.beam_size = int(cfg.get("beam_size", 1))
        self.device = device
        self.model = None

    async def load(self) -> None:
        # CTranslate2 dlopens libcublas.so at runtime. PyTorch bundles it, but 
        # doesn't always load it globally. We manually load it into the process
        # so CTranslate2 can find it regardless of LD_LIBRARY_PATH quirks.
        import ctypes
        import glob
        for lib_path in glob.glob("/opt/conda/lib/python3.10/site-packages/nvidia/*/lib/libcublas.so.12*"):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        for lib_path in glob.glob("/opt/conda/lib/python3.10/site-packages/nvidia/*/lib/libcudnn.so.*"):
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        
        from faster_whisper import WhisperModel

        device, _, index = self.device.partition(":")
        self.model = WhisperModel(
            self.model_name,
            device=device,
            device_index=int(index or 0),
            compute_type=self.compute_type,
        )

    async def warmup(self) -> None:
        session = _WhisperBatchSession(self)
        await session.accept(np.zeros(16000, dtype=np.float32))
        await session.final()

    def new_session(self) -> ASRSession:
        return _WhisperBatchSession(self)


class _WhisperBatchSession:
    def __init__(self, asr: FasterWhisperASR):
        self.asr = asr
        self.buffer: list[np.ndarray] = []

    async def accept(self, samples: np.ndarray) -> str | None:
        # Batch does no work during speech. This is the cost stream_all exists
        # to move out of the post-endpoint window.
        self.buffer.append(samples)
        return None

    async def final(self) -> str:
        audio = (
            np.concatenate(self.buffer) if self.buffer
            else np.zeros(0, dtype=np.float32)
        )

        def run() -> str:
            segments, _info = self.asr.model.transcribe(
                audio,
                language=self.asr.language,
                beam_size=self.asr.beam_size,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            # `segments` is a generator and transcribe() does no work until it
            # is consumed. Timing the call alone would record roughly zero.
            return "".join(s.text for s in segments).strip()

        return await asyncio.to_thread(run)


class WhisperStreamingASR:
    """Streaming ASR on whisper-streaming (LocalAgreement-2).

    Whisper has no streaming mode: it is an encoder-decoder trained on fixed 30s
    windows. whisper-streaming re-runs full inference on a growing buffer every
    `min_chunk_s` and commits the prefix two consecutive runs agree on, so the
    cadence is a policy knob rather than a model property.

    The win is not that less audio gets processed. Most of the decoding has
    already happened by t=0, leaving a short finalize instead of the whole
    utterance. The cost is several full passes per trial, on a GPU the LLM and
    TTS also want.

    Same Whisper weights as the batch backend. Only the policy differs, so the
    measured difference is attributable to the policy.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.model_name = cfg.get("model", "large-v3-turbo")
        self.compute_type = cfg.get("compute_type", "float16")
        self.language = cfg.get("language", "en")
        self.min_chunk_s = float(cfg.get("min_chunk_s", 1.0))
        self.device = device
        self.asr = None

    async def load(self) -> None:
        # whisper-streaming is a repo, not a package: clone ufal/whisper_streaming
        # onto PYTHONPATH. Its backend class shares a name with ours.
        from whisper_online import FasterWhisperASR as _Backend

        self.asr = _Backend(
            lan=self.language,
            modelsize=self.model_name,
            compute_type=self.compute_type,
        )

    async def warmup(self) -> None:
        session = self.new_session()
        frame = np.zeros(320, dtype=np.float32)
        for _ in range(int(2 * self.min_chunk_s * 50)):
            await session.accept(frame)
        await session.final()

    def new_session(self) -> ASRSession:
        from whisper_online import OnlineASRProcessor

        return _WhisperStreamSession(self, OnlineASRProcessor(self.asr))


class _WhisperStreamSession:
    """Runs inference on a worker task so frame intake never blocks.

    Inline inference would stop the cascade reading frames mid-pass, and that
    delay would land directly on t=0.
    """

    def __init__(self, asr: WhisperStreamingASR, online):
        self.online = online
        self.min_chunk_s = asr.min_chunk_s
        self.committed: list[str] = []
        self.pending: list[np.ndarray] = []
        self.pending_s = 0.0
        self.sr = 16000
        self.frames: asyncio.Queue = asyncio.Queue()
        self.partials: asyncio.Queue = asyncio.Queue()
        self.worker = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            samples = await self.frames.get()
            if samples is None:
                return
            self.pending.append(samples)
            self.pending_s += len(samples) / self.sr
            if self.pending_s < self.min_chunk_s:
                continue

            chunk = np.concatenate(self.pending)
            self.pending, self.pending_s = [], 0.0

            def step() -> str:
                self.online.insert_audio_chunk(chunk)
                out = self.online.process_iter()
                return (out[2] or "") if out else ""

            text = await asyncio.to_thread(step)
            if text:
                self.committed.append(text)
                self.partials.put_nowait(" ".join(self.committed).strip())

    async def accept(self, samples: np.ndarray) -> str | None:
        self.frames.put_nowait(samples)
        try:
            return self.partials.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def final(self) -> str:
        self.frames.put_nowait(None)
        await self.worker

        def flush() -> str:
            tail = self.online.finish()
            parts = list(self.committed)
            if tail and tail[2]:
                parts.append(tail[2])
            return " ".join(parts).strip()

        return await asyncio.to_thread(flush)


class MockASR:
    """Known delays, no model.

    `partial_every` emits a growing hypothesis every N frames, which is what
    gives stream_all a first-partial time to measure. Leave it at 0 and the
    session behaves like a batch backend that does nothing until the endpoint.
    """

    TRANSCRIPT = "what is the capital of france"

    def __init__(self, cfg: dict, device: str = "cpu"):
        d = cfg.get("delays_ms", {})
        self.final_ms = int(d.get("final", 300))
        self.partial_every = int(cfg.get("partial_every_frames", 0))

    async def load(self) -> None:
        return None

    async def warmup(self) -> None:
        return None

    def new_session(self) -> ASRSession:
        return _MockASRSession(self.final_ms, self.partial_every)


class _MockASRSession:
    def __init__(self, final_ms: int, partial_every: int):
        self.final_ms = final_ms
        self.partial_every = partial_every
        self.frames = 0

    async def accept(self, samples: np.ndarray) -> str | None:
        self.frames += 1
        if not self.partial_every or self.frames % self.partial_every:
            return None
        words = MockASR.TRANSCRIPT.split()
        return " ".join(words[: min(len(words), self.frames // self.partial_every)])

    async def final(self) -> str:
        await asyncio.sleep(self.final_ms / 1000)
        return MockASR.TRANSCRIPT


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

class VLLMEngine:
    """The LLM stage on vLLM.

    vLLM rather than transformers on purpose: continuous batching, prefix
    caching and KV cache behaviour decide what the concurrency sweep looks like,
    and they are properties of the serving stack rather than the model.

    It takes cuda:0 and offers no clean per-instance device argument, so place
    the other stages above it. `gpu_memory_utilization` defaults to 0.90, which
    leaves nothing for anything sharing the card.
    """

    def __init__(self, cfg: dict, device: str = "cuda:0"):
        self.model = cfg.get("model", "Qwen/Qwen3-4B-Instruct-2507")
        self.max_model_len = int(cfg.get("max_model_len", 2048))
        self.gpu_frac = float(cfg.get("gpu_memory_utilization", 0.90))
        self.dtype = cfg.get("dtype", "bfloat16")
        self.engine = None
        self.tokenizer = None

    async def load(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=self.model,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_frac,
                disable_log_stats=True,
            )
        )

    async def warmup(self) -> None:
        async for _ in self.generate("hello", "Answer in at most two sentences."):
            pass

    def _format(self, prompt: str, system: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            # Qwen3 emits a reasoning block unless this is off, which would blow
            # past max_tokens before a single spoken word appears.
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    async def generate(
        self, prompt: str, system: str, **params
    ) -> AsyncIterator[str]:
        import uuid

        from vllm import SamplingParams

        sampling = SamplingParams(
            temperature=float(params.get("temperature", 0.0)),
            seed=params.get("seed", 0),
            max_tokens=int(params.get("max_tokens", 96)),
        )
        emitted = 0
        stream = self.engine.generate(
            self._format(prompt, system), sampling, str(uuid.uuid4())
        )
        async for output in stream:
            # vLLM hands back cumulative text each step, not the delta.
            text = output.outputs[0].text
            if len(text) > emitted:
                yield text[emitted:]
                emitted = len(text)


class MockLLM:
    """Known delays, no model. The fixed response carries a sentence boundary
    partway through so the chunking policy has something to cut on."""

    RESPONSE = (
        "The capital of France is Paris. "
        "It has been the seat of government since the tenth century."
    )

    def __init__(self, cfg: dict, device: str = "cpu"):
        d = cfg.get("delays_ms", {})
        self.ttft_ms = int(d.get("first_token", 150))
        self.per_token_ms = int(d.get("per_token", 12))

    async def load(self) -> None:
        return None

    async def warmup(self) -> None:
        return None

    async def generate(
        self, prompt: str, system: str, **params
    ) -> AsyncIterator[str]:
        tokens = [w + " " for w in self.RESPONSE.split()]
        tokens = tokens[: int(params.get("max_tokens", 96))]
        await asyncio.sleep(self.ttft_ms / 1000)
        for i, tok in enumerate(tokens):
            if i:
                await asyncio.sleep(self.per_token_ms / 1000)
            yield tok


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------

def _use_soundfile_for_wavs() -> None:
    """Points torchaudio.load at soundfile.

    torchaudio 2.9 removed its own decoders and routes load() through
    torchcodec, ignoring the backend argument CosyVoice passes. torchcodec then
    has to match both the FFmpeg and the CUDA the container happens to have, and
    the wheel pip resolves here is built against a newer CUDA than torch is.

    CosyVoice only needs a wav read off disk, which soundfile already does.
    Patching the one function is smaller than pinning a chain of wheels against
    each other, and it keeps the TTS stage's audio loading identical to the
    harness's own.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def load(path, *args, **kwargs):
        # torchaudio returns (channels, frames); soundfile gives (frames, channels).
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T).contiguous(), sample_rate

    torchaudio.load = load


class CosyVoice2TTS:
    """The autoregressive codec-LM family.

    Streams by construction: the LM emits acoustic tokens continuously and the
    codec decodes them incrementally, so audio starts before the chunk is
    finished being synthesized.
    """

    family = "ar"

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.model_dir = cfg.get("model", "iic/CosyVoice2-0.5B")
        # Zero-shot needs a reference voice. Fixed across all trials: a
        # different speaker means different durations and different TTS work.
        self.prompt_wav = cfg.get("prompt_wav", "")
        self.prompt_text = cfg.get("prompt_text", "")
        self.sample_rate = 24000
        self.device = device
        self.model = None

    async def load(self) -> None:
        import torch

        from cosyvoice.cli.cosyvoice import CosyVoice2

        self.model = CosyVoice2(self.model_dir, load_jit=False, load_trt=False, fp16=False)
        # CosyVoice chooses its own device. A CPU fallback still produces audio,
        # just many times slower, so it has to be visible rather than looking
        # like a slow model.
        device = next(self.model.model.llm.parameters()).device
        print(f"cosyvoice2 on {device} (cuda available: {torch.cuda.is_available()}, "
              f"hop {self.model.model.token_hop_len})", flush=True)
        if device.type != "cuda":
            raise RuntimeError(f"cosyvoice2 loaded on {device}, every timing would be junk")

    async def warmup(self, stream: bool = False) -> None:
        async for _ in self.synth("Warming up the decoder.", stream=stream):
            pass

    async def synth(self, text: str, stream: bool = True) -> AsyncIterator[np.ndarray]:
        # CosyVoice splits the text and runs each piece through wetext. A piece
        # that normalizes to nothing trips an assertion inside the tokenizer, so
        # newlines and runs of whitespace are collapsed first: an LLM response
        # with a line break produces exactly that empty piece.
        text = " ".join(text.split())
        if not text:
            raise RuntimeError("cosyvoice2 got empty text")

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            # Blocking generator, so it runs off the event loop and pieces are
            # handed over as they appear rather than all at the end.
            try:
                # The reference goes in as a path: the frontend resamples it to
                # both 16k and 24k itself, so it does its own reading.
                gen = (
                    self.model.inference_zero_shot(
                        text, self.prompt_text, self.prompt_wav, stream=stream
                    )
                    if self.prompt_text
                    else self.model.inference_cross_lingual(
                        text, self.prompt_wav, stream=stream
                    )
                )
                for out in gen:
                    audio = out["tts_speech"].cpu().numpy().reshape(-1)
                    loop.call_soon_threadsafe(queue.put_nowait, audio)
            except BaseException as exc:  # noqa: BLE001
                # Onto the queue, not swallowed. A bare finally here would end
                # the stream cleanly on failure, and the trial would record a
                # successful synthesis of no audio.
                #
                # Re-raised with the text attached, because the failures that
                # come out of the text frontend are bare assertions with no
                # message and the input is the only clue.
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    RuntimeError(f"{type(exc).__name__}: {exc} | text={text!r}"),
                )
            else:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        import time

        started = time.monotonic()
        loop.run_in_executor(None, produce)
        produced = 0
        while True:
            piece = await queue.get()
            if isinstance(piece, BaseException):
                raise piece
            if piece is None:
                if not produced:
                    raise RuntimeError(f"cosyvoice2 returned no audio for {text!r}")
                return
            produced += 1
            print(f"  tts piece {produced}: {(time.monotonic() - started) * 1000:.0f}ms, "
                  f"{len(piece) / self.sample_rate:.2f}s audio, {len(text)} chars", flush=True)
            yield piece.astype(np.float32, copy=False)


class F5TTS:
    """The flow-matching family.

    Non-autoregressive: there is no token stream, because the model solves for
    the whole chunk at once. So "streaming" here means the client cuts the text
    smaller, not that the model produces output gradually. An AR model's first
    audio arrives mid-chunk; F5's cannot arrive until the chunk is complete.
    That asymmetry is the finding, not a defect in the grid.
    """

    family = "nar"

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.model_name = cfg.get("model", "F5TTS_v1_Base")
        self.ref_wav = cfg.get("prompt_wav", "")
        self.ref_text = cfg.get("prompt_text", "")
        self.nfe_steps = int(cfg.get("nfe_steps", 32))
        self.sample_rate = 24000
        self.device = device
        self.api = None

    async def load(self) -> None:
        from f5_tts.api import F5TTS as F5API

        self.api = F5API(model=self.model_name, device=self.device)

    async def warmup(self) -> None:
        async for _ in self.synth("Warming up the decoder."):
            pass

    async def synth(self, text: str, stream: bool = True) -> AsyncIterator[np.ndarray]:
        # Flow matching has nothing to stream, so the flag is accepted and
        # ignored. The client decides chunk size; the model cannot emit early.
        def run() -> np.ndarray:
            wav, sr, _ = self.api.infer(
                ref_file=self.ref_wav,
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=self.nfe_steps,
                remove_silence=False,
            )
            self.sample_rate = sr
            return np.asarray(wav, dtype=np.float32).reshape(-1)

        # One piece per chunk. Nothing to stream: the solver has to finish.
        yield await asyncio.to_thread(run)


class MockTTS:
    """Known delays, no model.

    Audio duration scales with text length so real-time factor and underruns
    mean something. A mock emitting a fixed amount regardless of input would
    make RTF meaningless and hide the failure the metric watches for.

    Pieces are 80ms, matching a 12.5Hz codec frame rate. That is the floor on
    streaming granularity for an AR family: however fast the GPU, audio cannot
    arrive in finer pieces than one codec frame.
    """

    family = "nar"
    PIECE_S = 0.08

    def __init__(self, cfg: dict, device: str = "cpu"):
        d = cfg.get("delays_ms", {})
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.first_piece_ms = int(d.get("first_chunk", 120))
        # Below 1.0 means synthesis outruns playback, which is what keeps the
        # stream from starving once it has started.
        self.rtf = float(cfg.get("rtf", 0.3))
        self.seconds_per_char = float(cfg.get("s_per_char", 0.06))

    async def load(self) -> None:
        return None

    async def warmup(self) -> None:
        async for _ in self.synth("warm"):
            pass

    async def synth(self, text: str, stream: bool = True) -> AsyncIterator[np.ndarray]:
        total_s = max(self.PIECE_S, len(text) * self.seconds_per_char)
        n = max(1, round(total_s / self.PIECE_S))
        samples = np.zeros(int(self.sample_rate * self.PIECE_S), dtype=np.float32)

        await asyncio.sleep(self.first_piece_ms / 1000)
        for i in range(n):
            if i:
                await asyncio.sleep(self.PIECE_S * self.rtf)
            yield samples


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

ASR_BACKENDS = {
    "faster-whisper": FasterWhisperASR,
    "whisper-streaming": WhisperStreamingASR,
    "mock": MockASR,
}
LLM_BACKENDS = {"vllm": VLLMEngine, "mock": MockLLM}
TTS_BACKENDS = {"cosyvoice2": CosyVoice2TTS, "f5": F5TTS, "mock": MockTTS}


def build(kind: str, cfg: dict, devices: dict):
    """Constructs one stage from its config block.

    `devices` maps stage name to a torch device string. Placement is recorded
    into every trace, so a one-GPU run and a three-GPU run can never be
    compared by accident.
    """
    table = {"asr": ASR_BACKENDS, "llm": LLM_BACKENDS, "tts": TTS_BACKENDS}[kind]
    name = cfg.get("backend", "mock")
    if name not in table:
        raise ValueError(f"{kind} backend must be one of {sorted(table)}")
    return table[name](cfg, devices.get(kind, os.environ.get("DEVICE", "cuda")))
