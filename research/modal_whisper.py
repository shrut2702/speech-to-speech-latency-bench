"""Whisper ASR, batch and streaming. A spike, not part of the harness.

    modal run research/modal_whisper.py --warmup
    modal run research/modal_whisper.py --warmup --mode stream
    modal run research/modal_whisper.py --warmup --mode both

Both paths load the same weights, so the only difference measured is policy.

Batch buffers the whole utterance and transcribes once at the endpoint, so its
cost lands entirely after the user stops talking. Streaming re-runs inference on
a growing buffer while they are still speaking and commits the prefix that two
consecutive runs agree on, leaving a short finalize at the endpoint. The number
worth comparing is what each still owes at the endpoint, not what each spends in
total.

Streaming is fed at wall clock pace, as the harness feeds it. Handing it the
whole array at once would make it indistinguishable from batch.
"""

import modal

MODEL = "large-v3-turbo"
STREAMING_REPO = "/opt/whisper_streaming"

image = (
    # cuDNN in the base image. CTranslate2 needs it and the slim images do not
    # ship it, which surfaces as a library load error rather than a clear one.
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git")
    .pip_install("faster-whisper>=1.0", "soundfile>=0.12", "librosa>=0.10")
    # whisper-streaming is a repo rather than a package.
    .run_commands(
        f"git clone --depth 1 https://github.com/ufal/whisper_streaming {STREAMING_REPO}"
    )
    .env({"HF_HOME": "/cache/hf", "PYTHONPATH": STREAMING_REPO})
    # Mounted, not baked: editing a clip should not rebuild the image.
    .add_local_dir("data/clips", "/root/clips")
    .add_local_file("data/manifest.jsonl", "/root/manifest.jsonl")
)

app = modal.App("whisper-spike", image=image)
cache = modal.Volume.from_name("s2s-models", create_if_missing=True)


@app.function(gpu="A10G", volumes={"/cache": cache}, timeout=3600)
def run(
    clip_ids: list[str], lengths: list[int], mode: str, min_chunk_s: float,
    warmup: bool,
) -> None:
    import itertools
    import json
    import time

    import numpy as np
    import soundfile as sf

    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in open("/root/manifest.jsonl", encoding="utf-8")
        if line.strip()
    }

    def speech_of(row):
        """Just the speech, without the trailing pad. That is what the batch
        path transcribes at the endpoint."""
        audio, sr = sf.read(f"/root/clips/{row['file']}", dtype="float32")
        return audio[: row["endpoint_sample"]], sr

    def stitched(seconds: int):
        """A clip of a given length, built by joining real speech end to end.

        The set tops out at 18.7s, and the interesting question is what happens
        past 30s, where Whisper stops fitting an utterance in one encoder window
        and starts processing several. Nothing shorter can answer that.
        """
        parts, total, sr = [], 0.0, 16000
        for row in itertools.cycle(rows.values()):
            audio, sr = speech_of(row)
            parts.append(audio)
            total += len(audio) / sr
            if total >= seconds:
                break
        return np.concatenate(parts)[: int(seconds * sr)], sr

    clips = [(cid, *speech_of(rows[cid]), rows[cid]["bucket"]) for cid in clip_ids]
    clips += [(f"stitched_{s}s", *stitched(s), "synthetic") for s in lengths]

    # ---- batch ----------------------------------------------------------

    def batch(audio, model):
        """Times a full transcription.

        `segments` is a generator and transcribe() does no work until it is
        consumed, so the join is inside the timed region. Timing the call alone
        would record roughly zero.
        """
        t0 = time.monotonic()
        segments, _info = model.transcribe(
            audio, language="en", beam_size=1,
            condition_on_previous_text=False, vad_filter=False,
        )
        text = "".join(s.text for s in segments).strip()
        return text, (time.monotonic() - t0) * 1000, 1

    # ---- streaming ------------------------------------------------------

    def stream(audio, sr, asr):
        """Feeds at wall clock pace and times only what is left at the endpoint.

        The finalize clock starts at the end of speech, not after the last
        incremental pass, because a live system reaches the endpoint whether or
        not the previous pass has returned.
        """
        from whisper_online import OnlineASRProcessor

        online = OnlineASRProcessor(asr)
        step = int(min_chunk_s * sr)
        committed, passes, busy_ms = [], 0, 0.0

        t0 = time.monotonic()
        end_of_speech = t0 + len(audio) / sr
        first_partial_ms = None

        # Only whole chunks run during speech. The remainder is left for the
        # finalize on purpose: a live system reaches the endpoint partway
        # through a chunk, and decoding that leftover is the cost it pays.
        start = 0
        while start + step <= len(audio):
            due = t0 + (start + step) / sr
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)

            online.insert_audio_chunk(audio[start : start + step])
            t = time.monotonic()
            out = online.process_iter()
            busy_ms += (time.monotonic() - t) * 1000
            passes += 1
            if out and out[2]:
                committed.append(out[2])
                if first_partial_ms is None:
                    first_partial_ms = (time.monotonic() - t0) * 1000
            start += step

        if start < len(audio):
            online.insert_audio_chunk(audio[start:])

        # The clock starts at the endpoint, not when the last pass returned. A
        # live system reaches the endpoint whether or not it has caught up.
        t = max(end_of_speech, time.monotonic())
        while time.monotonic() < t:
            time.sleep(0.001)

        # One more pass, then the flush. process_iter is what runs inference;
        # finish() only drains the hypothesis buffer. Calling finish() alone
        # leaves the audio since the last chunk boundary undecoded, which reads
        # as a 0ms finalize and a truncated transcript.
        out = online.process_iter()
        if out and out[2]:
            committed.append(out[2])
        tail = online.finish()
        finalize_ms = (time.monotonic() - t) * 1000
        busy_ms += finalize_ms
        passes += 1

        if tail and tail[2]:
            committed.append(tail[2])
        return " ".join(committed).strip(), finalize_ms, passes, busy_ms, first_partial_ms

    # ---- go -------------------------------------------------------------

    if mode in ("batch", "both"):
        from faster_whisper import WhisperModel

        model = WhisperModel(MODEL, device="cuda", compute_type="float16")
        if warmup:
            batch(np.zeros(16000, dtype=np.float32), model)
            print("batch warmed", flush=True)

        print(f"\nbatch\n{'clip':18s} {'speech':>8s} {'at endpoint':>12s} "
              f"{'passes':>7s} {'windows':>8s}")
        for label, audio, sr, _bucket in clips:
            seconds = len(audio) / sr
            text, ms, passes = batch(audio, model)
            print(f"{label:18s} {seconds:7.2f}s {ms:11.0f}ms {passes:7d} "
                  f"{-(-int(seconds) // 30):8d}")
            print(f"   {text[:90]}")

    if mode in ("stream", "both"):
        from whisper_online import FasterWhisperASR as StreamingBackend

        asr = StreamingBackend(lan="en", modelsize=MODEL)
        if warmup:
            stream(np.zeros(int(3 * 16000), dtype=np.float32), 16000, asr)
            print("stream warmed", flush=True)

        print(f"\nstreaming, min_chunk={min_chunk_s}s\n{'clip':18s} {'speech':>8s} "
              f"{'at endpoint':>12s} {'passes':>7s} {'gpu busy':>9s} {'1st partial':>12s}")
        for label, audio, sr, _bucket in clips:
            seconds = len(audio) / sr
            text, ms, passes, busy, first = stream(audio, sr, asr)
            first_s = "-" if first is None else f"{first:.0f}ms"
            print(f"{label:18s} {seconds:7.2f}s {ms:11.0f}ms {passes:7d} "
                  f"{busy:8.0f}ms {first_s:>12s}")
            print(f"   {text[:90]}")


@app.local_entrypoint()
def main(
    clips: str = "llamaq_0014,gsm8k_0000,gsm8k_0004",
    # The real clips all fit inside one 30s encoder window, so they cannot show
    # what happens past it.
    lengths: str = "40,80",
    mode: str = "batch",
    # How often streaming re-runs inference. Smaller leaves less to finalize at
    # the endpoint and costs more passes on a GPU the LLM and TTS also want.
    min_chunk_s: float = 1.0,
    warmup: bool = False,
):
    run.remote(
        [c.strip() for c in clips.split(",") if c.strip()],
        [int(x) for x in lengths.split(",") if x.strip()],
        mode,
        min_chunk_s,
        warmup,
    )
