# s2s-latency-bench

A harness for measuring where the milliseconds go in speech-to-speech systems, and a comparison of a cascaded pipeline (ASR to LLM to TTS) against a full-duplex model (Moshi).

Status: infrastructure in place, model integrations in progress.

---

## Why another benchmark

Most latency numbers for speech systems are measured by loading a wav, calling the model, and timing the call. That measures throughput, not latency. A conversational system cannot see the future: audio arrives at 1x speed and it has to decide what to do with each frame as it lands. Hand it the whole utterance at once and streaming ASR becomes indistinguishable from batch ASR, because both get everything instantly.

So this harness streams audio at wall-clock pace, 20ms at a time, exactly like a microphone. Every system under test consumes the identical stream.

## Where the clock starts

`t = 0` is the annotated end of user speech, taken from the manifest and converted to a scheduled instant.

It is deliberately not a VAD decision. A VAD threshold contributes a constant you already know (set it to 300ms or 700ms and every measurement shifts by 400ms) plus per-clip variance that depends on how the speaker trailed off, their accent, and background noise. You would be hunting 40ms differences underneath 200ms of endpointing jitter.

It also would not be fair. Moshi is full-duplex and has no endpointing step at all, so bolting a VAD onto the cascade alone rigs the comparison, and bolting it onto both penalizes Moshi for a component its architecture deliberately removes.

Using the annotated endpoint means t=0 is a property of the audio file rather than of any system: identical across configs, deterministic across runs. A production deployment adds roughly 200 to 700ms of endpointing on top, and that belongs in the writeup as a stated constant rather than inside the measurement.

## What gets compared

Three cascade paths, increasing in how much overlaps:

| Config | ASR | LLM to TTS |
|---|---|---|
| `cascade_batch` | waits for the full utterance | fully sequential |
| `cascade_stream_gen` | waits for the full utterance | LLM streams, TTS starts on sentence 1 |
| `cascade_stream_all` | runs during speech | LLM streams, TTS starts on sentence 1 |

Plus `moshi` as the full-duplex baseline.

Two notes on fairness. The streaming ASR config runs the **same Whisper weights** as the batch config, so the only difference is the streaming policy and the delta is attributable to it. And "streaming" means different things per TTS family: an AR codec-LM emits acoustic tokens continuously and the codec decodes incrementally, while a flow-matching model has nothing to stream and gets chunked on sentence boundaries instead. That asymmetry is a finding, not a defect in the grid.

## Metrics

Headline is time to first audio, measured from the endpoint, reported as p50/p95/p99. Never means.

Alongside it: per-stage first-output (ASR finalize, LLM first token, TTS first chunk), real-time factor, inter-chunk gaps and underruns, and behaviour at 1, 2, 4 and 8 concurrent sessions.

Two validity gates run automatically. `check_work_constant` fails the report if the LLM produced different response lengths across configs, since then the configs did different amounts of work and the latencies are not comparable. And any trial where the feeder fell more than 50ms behind schedule is discarded, because it describes a loaded host rather than a pipeline.

## Usage

```bash
pip install -r requirements.txt

# trim, pad to a uniform tail, loudness-normalize, annotate the endpoint
python scripts/prepare_clips.py --src raw/ --out data/clips --manifest data/manifest.jsonl

python -m bench.runner --config configs/cascade_batch.yaml
python -m bench.runner --config configs/cascade_stream_gen.yaml
python -m bench.runner --config configs/cascade_stream_all.yaml
python -m bench.runner --config configs/moshi.yaml

python scripts/report.py results/*.jsonl
```

Traces are committed; the tables regenerate from them.

## Clips

Audio is not committed, for licensing reasons. `prepare_clips.py` takes whatever you point it at and produces the normalized set plus manifest.

The set used here draws on the human-recorded subsets of [VoiceBench](https://github.com/matthewcym/voicebench) (`wildvoice`, `commoneval`), with [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) for turn-taking and barge-in. TTS-generated clips are avoided for anything touching quality: ASR finds them unrealistically easy, which erases the error-propagation failure mode that is the cascade's main structural weakness.

Voice-assistant corpora are almost entirely short queries, so a handful of 10 to 15 second utterances are added separately. That bucket matters more than it looks: batch ASR cost scales with utterance length while streaming ASR barely moves, so without long clips the streaming win looks unimpressive and you draw the wrong conclusion.

## Reading the results honestly

The cascade's answer quality is its LLM's quality, so swapping the 4B for an 8B moves it while Moshi stays put. Nothing here measures an intrinsic property of "cascades" versus "end-to-end". The useful output is a latency-quality frontier across several cascade configurations and Moshi, not a two-row table.

All cascade stages share one GPU, so they contend for SMs and bandwidth in a way separate services would not. This hurts the streaming paths most, since overlapping only pays off when stages genuinely run in parallel, which means the reported streaming win is a conservative estimate. Moshi is a single model and has no such contention, so the setup mildly disadvantages the cascade.

## Layout

```
bench/
  feeder.py       wall-clock audio streaming, the core of the methodology
  trace.py        event log; metrics are derived, never computed inline
  metrics.py      TTFA, stage breakdown, streaming health, validity gates
  runner.py       config in, traces out
  systems/        cascade.py, moshi.py behind one interface
scripts/
  prepare_clips.py
  report.py
configs/          one yaml per system variant
data/             manifest plus prepared clips
results/          committed traces
```
