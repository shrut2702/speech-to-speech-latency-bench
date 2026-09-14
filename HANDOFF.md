# Handoff

Context for picking this project up cold.

## What this is and why it exists

A harness that measures where the milliseconds go in speech-to-speech systems, comparing a cascaded pipeline (ASR to LLM to TTS) against Moshi (full-duplex, end to end).

It exists to target a specific job. Flam's AI Engineer JD has the line *"Own latency. Profile the pipeline, find where the milliseconds go, and remove them,"* plus *"build evaluation harnesses that tell us something true"* and *"read the paper, replicate it, decide honestly whether it's worth shipping."* This project is that brief, delivered as an artifact.

It also closes the biggest gap against that JD: no hands-on experience with a modern inference stack. The cascade's LLM stage runs on vLLM deliberately, so continuous batching, prefix caching and KV cache behaviour get measured rather than read about.

Secondary constraint that shaped everything: the author's public portfolio is all notebooks, and the JD says "not only in notebooks." So this ships as a repo with a CLI, YAML configs, a Dockerfile and committed traces, not as a notebook.

## Design decisions, and why (do not silently undo these)

**Audio is streamed at wall-clock pace, 20ms frames, never fed as whole files.**
Loading a wav and timing the call measures throughput, not latency. It also makes streaming ASR indistinguishable from batch ASR, since both get all the audio instantly. `bench/feeder.py` is the core of the methodology for this reason.

**t = 0 is the manifest's annotated end-of-speech, not a VAD decision.**
Three reasons. A VAD threshold contributes a constant you already know (300ms vs 700ms shifts every measurement by 400ms) plus per-clip variance from how the speaker trailed off, so you would be hunting 40ms differences under 200ms of jitter. Moshi is full-duplex and has no endpointing at all, so adding a VAD to the cascade alone rigs the comparison and adding it to both penalizes Moshi for a component it deliberately removes. And an annotated endpoint is a property of the audio file, so it is identical across configs and deterministic across runs.
Report endpoint delay as a stated constant in the writeup instead (a deployment adds roughly 200 to 700ms).

**Systems react at the endpoint, never at end-of-stream.**
The file running out is an out-of-band signal a live system never receives. Waiting for it silently adds the whole trailing-silence pad to every number. This bug was already made once and caught by the mock: TTFA read 1069ms when the delays summed to 270ms. `drain_to_endpoint()` in `bench/systems/base.py` is the fix. Keep consuming the tail in the background, because streaming ASR needs trailing context to finalize.

**Streaming ASR uses the same Whisper weights as the batch config.**
Only the policy differs (`whisper_streaming`, local-agreement), so the measured delta is attributable to streaming rather than to a different model. Using a natively streaming model as the primary would change two variables at once.

**The LLM is pinned: temperature 0, fixed seed, response length capped by system prompt.**
If response length varies, TTS does different amounts of work and the configs are not comparable. `check_work_constant` in `metrics.py` fails the report automatically if this drifts.

**Percentiles only, never means.** Warmup trials are discarded. Trials where the feeder fell more than 50ms behind schedule are discarded as describing a loaded host.

**The 3 paths x 2 TTS families grid is not orthogonal, on purpose.**
An AR codec-LM streams acoustic tokens and the codec decodes incrementally. A flow-matching model has no tokens to stream, so streaming means sentence chunking. Say this in the writeup rather than presenting a clean grid. The open question worth answering: does token-streaming an AR model actually beat sentence-chunking a fast NAR one like Kokoro?

## State

Working and validated:

- `bench/feeder.py` wall-clock streaming, absolute deadlines so no drift, deterministic endpoint
- `bench/trace.py` monotonic event log to JSONL, `mark_at()` for computed instants
- `bench/metrics.py` TTFA, per-stage, RTF, inter-chunk gaps, underruns, percentiles, validity gates
- `bench/runner.py` config in, traces out, records GPU name in every trace
- `bench/systems/base.py` the `S2SSystem` interface plus `drain_to_endpoint()`
- `bench/systems/mock.py` synthetic system with known delays
- `scripts/prepare_clips.py` trim, pad to uniform tail, loudness normalize, annotate endpoint
- `scripts/report.py` regenerates all tables from committed traces
- `tests/test_feeder.py` 5 tests, all passing

Verification run: `mock_fast` expected 270ms TTFA, measured 292ms p50; `mock_slow` expected 900ms, measured 936ms. The ~25ms residual is asyncio scheduling plus 20ms frame granularity, since a system can only react at the frame boundary after the endpoint. Consistent across configs, so the measurement layer is sound.

Stubs with TODOs at each model call:

- `bench/systems/cascade.py` needs faster-whisper (batch), whisper_streaming (stream_all), vLLM + Qwen3-4B, and a TTS (CosyVoice2 for AR, F5-TTS or Kokoro for NAR)
- `bench/systems/moshi.py` needs kyutai-labs/moshi plus the Mimi codec

## Next steps, in order

1. **Check Moshi's real-time factor on the target GPU before anything else.** If RTF exceeds 1, every number describes the hardware rather than the architecture. Rent an A100 40GB for measurement runs if needed. A100 is the pick since FP8 is not required (quantization is explicitly out of scope for now, so Ada is unnecessary).
2. ~~Prepare real clips.~~ **Done.** `scripts/fetch_voicebench.py` pulls the human subsets and `prepare_clips.py` normalizes them. Current set: 15 clips, 5 each in short/medium/long, 2.6s to 11.7s, all with reference transcripts so ASR WER and error propagation are measurable. Still worth adding Full-Duplex-Bench clips for turn-taking and barge-in before the Moshi comparison.
3. Fill `cascade.py` for `path: batch` only. Get one config producing real audio end to end before touching the others.
4. Add `stream_gen`, then `stream_all`.
5. Wire Moshi.
6. Concurrency sweep at 1, 2, 4, 8.

## Things to know before running

Runs take real wall-clock time by design, because the feeder streams at 1x. Three clips at 20s each, 5 trials, 2 configs is roughly 10 minutes. Budget accordingly for the concurrency sweeps.

All cascade stages share one GPU, so they contend in a way separate services would not. This hurts the streaming paths most, since overlapping only pays off when stages genuinely run in parallel, which makes the reported streaming win a conservative estimate. Moshi is a single model and has no such contention, so the setup mildly disadvantages the cascade. Measure the contention cheaply first (stage timings in isolation vs in-pipeline, plus GPU utilization) before considering a second GPU.

Single-session contention is LLM against TTS against vocoder, not ASR against LLM, because a half-duplex cascade's LLM waits for the final transcript. ASR contention only appears at concurrency above 1.

## Planned sequel

Add DINet as a stage 5 avatar output in the same repo, turning "speech-to-speech latency" into "speech-to-avatar latency." The finding to hunt: DINet uses a centered audio window, so frame *t* needs audio from roughly *t+2*, which structurally forces video to lag audio by about 80ms. Either ship desynced output or delay the audio and eat the latency. Measure it rather than just mentioning it. MuseTalk is the pragmatic swap if streaming DINet fights back, since the deliverable is the system and the measurements, not the model.
