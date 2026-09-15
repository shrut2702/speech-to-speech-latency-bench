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

| Path | ASR | LLM to TTS |
|---|---|---|
| `batch` | waits for the full utterance | fully sequential |
| `stream_gen` | waits for the full utterance | LLM streams, TTS starts on chunk 1 |
| `stream_all` | runs during speech | LLM streams, TTS starts on chunk 1 |

Crossed with two TTS families, giving six cascade configs plus Moshi:

|  | CosyVoice2 (AR) | F5-TTS (NAR) |
|---|---|---|
| `batch` | `cascade_batch_cosyvoice2` | `cascade_batch_f5` |
| `stream_gen` | `cascade_stream_gen_cosyvoice2` | `cascade_stream_gen_f5` |
| `stream_all` | `cascade_stream_all_cosyvoice2` | `cascade_stream_all_f5` |

Each config names its TTS family, and the cascade refuses to run if the loaded backend reports a different one. Without that check a config could carry one family's name over the other family's numbers, which would invert the comparison the grid exists for.

Plus `moshi` as the full-duplex baseline.

Two notes on fairness. The streaming ASR config runs the **same Whisper weights** as the batch config, so the only difference is the streaming policy and the delta is attributable to it. And "streaming" means different things per TTS family: an AR codec-LM emits acoustic tokens continuously and the codec decodes incrementally, while a flow-matching model has nothing to stream and gets chunked on sentence boundaries instead. That asymmetry is a finding, not a defect in the grid.

## One process, several GPUs

Every stage is an object in one process, placed on its own card by a `devices:` block in the config. There is no service layer, no wire protocol and nothing to launch.

The alternative was a stage per service, which is how you would deploy this for real. It buys dependency isolation and costs a network hop inside every stage number. A benchmark is a job rather than a production system, so the hop is pure measurement error and the isolation is a problem to solve only if the torch pins actually collide. If one day they do, the fix is a subprocess with its own venv for the offending stage, not a service mesh.

vLLM takes `cuda:0` and offers no clean per-instance device argument, so the other stages sit above it. Placement is recorded into every trace, so a one-GPU run and a three-GPU run can never be compared by accident.

The feeder stays in the harness throughout. It is simulating the microphone, so handing a stage a file path or a whole array would put us back to measuring throughput.

## Metrics

Percentiles only, p50/p95/p99. Never means: the tail is what a user notices.

Everything is measured twice, against two different clocks, because "how long after the user stopped talking" and "how long did that stage take" are different questions and only the first accumulates the stages before it.

**From the endpoint** — what the user sits through:

| metric | span |
|---|---|
| time to first audio | endpoint → first audio out |
| end to end | endpoint → last audio out |
| asr final | endpoint → final transcript |
| llm first token | endpoint → first token |
| tts first chunk | endpoint → first synthesized chunk |

**From each stage's own start** — what that stage costs, with everything before it subtracted out. This is the view that says which millisecond to go delete:

| metric | span | note |
|---|---|---|
| asr first partial | ASR start → first partial | streaming only; batch produces none |
| asr total | ASR start → final transcript | batch starts at the endpoint, streaming at the first frame, so this reads as transcription cost for one and wall-clock-since-speech for the other |
| llm ttft | LLM start → first token | dominated by prefill |
| llm 2nd token | first token → second token | one decode step with a warm KV cache, which is the only honest read on per-token cost |
| llm total | LLM start → last token | |
| tts first audio | TTS start → first audio out | |
| tts total | TTS start → last audio out | |

**AR TTS internals**, for the codec-LM family only: time to first acoustic token, time for the whole token stream, and the decoder's time to first chunk and to the whole response. The event names exist and `report.py` prints the table when they appear, but nothing emits them yet: separating CosyVoice2's LM from its flow decoder and vocoder needs internals rather than its public API, which streams them together. Until that is wired the column is empty rather than guessed at.

Alongside all of it: real-time factor, inter-chunk gaps, underruns, and behaviour at 1, 2, 4 and 8 concurrent sessions.

## What each trial keeps

A latency number without the output that produced it cannot be checked for whether speed cost accuracy, so every trial stores what it actually made.

Text rides in the trace itself, since that is already one JSON record per trial:

- `artifacts.transcript` — what ASR heard, scored against the manifest's reference text
- `artifacts.response` — the LLM's full response
- `artifacts.tts_chunks` — the chunks handed to TTS, in order. Chunk one sets time to first audio, so this is what to read when that number moves. Batch has a single chunk and it is the whole response.

Audio is written separately when `audio_dir` is set, one directory per trial:

```
audio/<config>/<clip_id>/trial<N>/chunk_000.wav   as it arrived
                                 chunk_001.wav
                                 full.wav          the whole response
```

For the streaming paths the per-chunk files are the point: they are what the inter-chunk gap and underrun metrics are measuring, made listenable. Batch produces one chunk, so the two files hold the same audio.

Both go to the Modal results Volume, which `modal_app.py` commits after every config so a crash keeps whatever already finished. The write happens after the trial ends, never inside it — a disk write mid-stream would land in the gaps being measured.

Two validity gates run automatically. `check_work_constant` fails the report if the LLM produced different response lengths across configs, since then the configs did different amounts of work and the latencies are not comparable. And any trial where the feeder fell more than 50ms behind schedule is discarded, because it describes a loaded host rather than a pipeline.

## Usage

```bash
pip install -r requirements.txt

# download the clip set, stratified by length, with reference text
python scripts/fetch_clips.py --out raw

# trim, pad to a uniform tail, loudness-normalize, annotate the endpoint
python scripts/prepare_clips.py --src raw --out data/clips \
    --manifest data/manifest.jsonl --refs raw/refs.json

# check the harness against known delays. No GPU needed.
python -m bench.runner --config configs/cascade_mock.yaml

# the grid, one config at a time
for c in configs/cascade_*_*.yaml configs/moshi.yaml; do
    python -m bench.runner --config "$c"
done

python scripts/report.py results/*.jsonl
```

Or the whole sweep on Modal as one job, which is what `modal_app.py` is for:

```bash
modal run modal_app.py
```

One container with several GPUs, not a function per stage: separate functions land in separate containers on separate machines, and that hop would sit inside the measurement. Weights cache to a Volume so they download once, traces are committed to a Volume per config so a crash keeps what already finished, and the timeout is raised because the feeder streams at 1x.

Traces are committed; the tables regenerate from them.

## Clips

Audio is not committed, for licensing reasons. `fetch_clips.py` pulls the set from HuggingFace and `prepare_clips.py` normalizes it, so both regenerate from scratch.

45 clips from three sources, 15 each:

| source | short | medium | long | what it is |
|---|---|---|---|---|
| [`llama-questions`](https://huggingface.co/datasets/fixie-ai/llama-questions) | 10 | 5 | 0 | general knowledge, one-word answers |
| [`URO-Bench/MLCpro-en`](https://huggingface.co/datasets/Honggao/URO-Bench) | 5 | 6 | 4 | arithmetic and science, sentence answers |
| [`URO-Bench/Gsm8kEval`](https://huggingface.co/datasets/Honggao/URO-Bench) | 0 | 4 | 11 | multi-step word problems, worked answers |

The per-source quotas are lopsided because no single source spans the range: llama-questions tops out below 5s, Gsm8kEval never drops below 4s, and MLCpro-en holds only four clips past 8s. Skewing the quotas is what makes the combined set come out at 15 per bucket, which is the distribution that actually matters.

The long bucket matters more than it looks. Batch ASR decode cost grows with utterance length while streaming ASR barely moves, so a set of short queries makes the streaming win look unimpressive and you draw the wrong conclusion.

Every clip carries the reference text on both sides: `transcript` is what was asked, `reference_answer` is what a correct response says. Latency alone cannot tell you whether a faster path got quieter about being wrong, and error propagation, where an ASR mistake becomes a wrong answer, is the cascade's main structural weakness. The Gsm8kEval rows add `reference_short`, the bare final number, which scores by string match instead of needing a judge model.

One caveat to carry into any quality claim: this audio is synthesized. GSM8K is a text-only dataset and MLCpro-en is generated prompts, so neither has human recordings behind it. ASR finds synthetic speech easier than real microphone input, so measured WER reads optimistically low and the error-propagation effect is damped. Timing is unaffected, since the feeder only cares about duration and pacing.

## Reading the results honestly

The cascade's answer quality is its LLM's quality, so swapping the 4B for an 8B moves it while Moshi stays put. Nothing here measures an intrinsic property of "cascades" versus "end-to-end". The useful output is a latency-quality frontier across several cascade configurations and Moshi, not a two-row table.

Stage placement changes the answer, which is why `devices:` is recorded in every trace. Put the LLM and TTS on one card and they contend for SMs and bandwidth exactly when the streaming paths need them to overlap, so the streaming win reads lower than it should. Spread them and the cascade gets more hardware per session than Moshi, which runs as a single model on one card. Neither is the honest setup on its own: report GPUs per session alongside latency, and at concurrency above 1 report sessions per GPU at a latency target, which collapses both into one comparable number.

## Layout

```
bench/
  feeder.py           wall-clock audio streaming, the core of the methodology
  chunking.py         small first chunk, sentence-sized after
  trace.py            event log; metrics are derived, never computed inline
  metrics.py          TTFA, stage breakdown, streaming health, validity gates
  runner.py           config in, traces out
  stages.py           asr, llm and tts backends, plus mocks with known delays
  systems/            cascade.py, moshi.py behind one interface
modal_app.py          the whole sweep as one Modal job
scripts/
  fetch_clips.py      download the set from HuggingFace, stratified by length
  prepare_clips.py
  report.py
configs/              one yaml per system variant
data/                 manifest plus prepared clips
results/              committed traces
```
