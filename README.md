# s2s-latency-bench

Measures where the milliseconds go between a person finishing their sentence and hearing a reply. A cascaded pipeline (ASR, LLM, TTS) in three arrangements, against Moshi, which is full duplex.

Audio is fed at wall clock pace in 20ms frames, the way a microphone delivers it. `t = 0` is the annotated end of speech carried in the manifest, so it is a property of the audio file rather than of any system, and it is identical across configs and runs. A production deployment adds 200 to 700ms of endpointing on top of every number here.

Status: the cascade runs on Modal. F5-TTS and Moshi are not wired up yet.

## Configs

Three cascade paths, differing in how much of the work overlaps:

| path | ASR | LLM and TTS |
|---|---|---|
| `batch` | waits for the whole utterance | one after the other |
| `stream_gen` | waits for the whole utterance | LLM streams, TTS starts on chunk one |
| `stream_all` | transcribes during speech | LLM streams, TTS starts on chunk one |

Crossed with two TTS families, which is where the interesting asymmetry lives. An autoregressive codec LM can emit acoustic tokens as it goes. A flow matching model cannot start early at all, so "streaming" for it means chunking the text and calling it repeatedly.

|  | CosyVoice2 (autoregressive) | F5-TTS (flow matching) |
|---|---|---|
| `batch` | `cascade_batch_cosyvoice2` | `cascade_batch_f5` |
| `stream_gen` | `cascade_stream_gen_cosyvoice2` | `cascade_stream_gen_f5` |
| `stream_all` | `cascade_stream_all_cosyvoice2` | `cascade_stream_all_f5` |

Plus `moshi`, one model doing the whole job.

Behind them: faster-whisper `large-v3-turbo` for batch ASR, [whisper-streaming](https://github.com/ufal/whisper_streaming) over the same weights for `stream_all`, Qwen3-4B-Instruct on vLLM at temperature 0, and CosyVoice2-0.5B or F5-TTS.

## Three processes, three GPUs

Each stage gets its own process and its own card. Python threads share one interpreter lock, so in a single process the LLM and the TTS would take turns instead of running together, and overlapping them is the entire point of the streaming paths. Profiling CosyVoice2 measured that cost directly: its own LM went from 15.6 to 23.6 ms per token with a decoder thread alive beside it.

Every config uses the same three processes, including `batch` where nothing overlaps. If `batch` ran in one process and `stream_gen` did not, the gap between them would include the process boundary instead of just the overlap.

A worker sets `CUDA_VISIBLE_DEVICES` before torch is imported, so it sees one card and calls it `cuda:0`. The TTS worker also runs under its own venv, because CosyVoice needs numpy 1 and vLLM needs numpy 2. Placement is recorded in every trace, so a one-GPU run and a three-GPU run cannot be compared by accident.

## Clips

Audio is not committed, for licensing reasons. `fetch_clips.py` pulls the set from HuggingFace and `prepare_clips.py` normalizes it, so both regenerate from scratch.

45 clips from three sources, 15 each:

| source | short | medium | long | what it is |
|---|---|---|---|---|
| [`llama-questions`](https://huggingface.co/datasets/fixie-ai/llama-questions) | 10 | 5 | 0 | general knowledge, one-word answers |
| [`URO-Bench/MLCpro-en`](https://huggingface.co/datasets/Honggao/URO-Bench) | 5 | 6 | 4 | arithmetic and science, sentence answers |
| [`URO-Bench/Gsm8kEval`](https://huggingface.co/datasets/Honggao/URO-Bench) | 0 | 4 | 11 | multi-step word problems, worked answers |

The per-source quotas are lopsided because no single source covers the range. llama-questions tops out below 5s, Gsm8kEval never drops below 4s, and MLCpro-en holds only four clips past 8s. Skewing the quotas is what makes the combined set come out at 15 per length bucket.

The long bucket carries more weight than it looks. Batch ASR decode cost grows with utterance length while streaming ASR barely moves, so a set of short queries makes the streaming win look unimpressive and you draw the wrong conclusion.

Every clip carries reference text on both sides. `transcript` is what was asked, `reference_answer` is what a correct reply says, and Gsm8kEval rows add `reference_short`, the bare final number, which scores by string match instead of needing a judge model. Latency alone cannot tell you whether a faster path got quieter about being wrong.

One caveat for any quality claim: this audio is synthesized. GSM8K is text only and MLCpro-en is generated prompts, so neither has human recordings behind it. ASR finds synthetic speech easier than a real microphone, so measured WER reads optimistically low. Timing is unaffected, since the feeder only cares about duration and pacing.

## Reference voice

Both TTS backends clone zero shot, so they need a voice to imitate. `assets/prompt.wav` is 2.94s from VoiceBench wildvoice (`bc81b9bf75a1635cd60ee8bccf6ef063`), committed alongside its transcript in `assets/prompt.json`, and used by every config so that TTS work stays comparable across them.

It is trimmed. The original carried 1.18s of leading and 1.02s of trailing silence, and CosyVoice infers speaking rate from the prompt, so it read the speaker as slow and stretched every response to match. One 150-character answer came back as 32.8 seconds of audio, which is the model's token ceiling rather than a sentence.

## What gets measured

Percentiles only, p50 and p95. Means hide the tail, and the tail is what a user notices.

Everything is timed against two clocks. **From the endpoint** is what the user sits through, and it accumulates every stage before it: time to first audio, end to end, final transcript, first LLM token, first synthesized chunk. **From each stage's own start** is what that stage costs on its own, which is the view that says which millisecond to go delete: ASR first partial and total, LLM time to first token, second token and total, TTS time to first audio and total.

Alongside those: real time factor, inter-chunk gaps, and underruns.

Two gates run automatically. The report fails if the LLM produced different response lengths across configs, since then the configs did different amounts of work. And any trial where the feeder fell more than 50ms behind schedule is thrown out, because it describes a loaded host rather than a pipeline.

## What gets stored

A latency number without the output that produced it cannot be checked for whether speed cost accuracy, so every trial keeps what it made.

Text rides in the trace, which is already one JSON record per trial: the transcript ASR heard, the LLM's full response, and the chunks handed to TTS in order. Chunk one sets time to first audio, so that list is what to read when the number moves.

Audio is written per trial, one wav per chunk plus the whole response:

```
audio/<clip_id>/trial<N>/chunk_000.wav   as it arrived
                         chunk_001.wav
                         full.wav
```



## Usage

You need a Modal account and a HuggingFace token. The models install inside the container, so nothing on your own machine needs CUDA.

```bash
pip install -r requirements.txt
modal setup
modal secret create huggingface HF_TOKEN=hf_...
```

The secret has to carry that name, since `modal_app.py` asks for it by name and the run fails at startup without it. `requirements-models.txt` lists what ends up in the container, but `modal_app.py` is what builds it.

Then prepare the clips. This is not optional before a Modal run: `data/` is mounted from your working copy, so the wavs have to exist locally first.

```bash
# download the clip set, stratified by length, with reference text
python scripts/fetch_clips.py --out raw

# trim, pad, loudness-normalize, annotate the endpoint
python scripts/prepare_clips.py --src raw --out data/clips \
    --manifest data/manifest.jsonl --refs raw/refs.json

# check the harness against known delays, no GPU needed
python -m bench.runner --config configs/cascade_mock.yaml
```

The real runs go to Modal, three A100s in one container:

```bash
modal run modal_app.py --configs cascade_batch_cosyvoice2 --sample-only
modal run modal_app.py                                    # the whole sweep
```

`--sample-only` is one clip, one trial, for shaking a config out. Weights cache to a Volume so they download once.


