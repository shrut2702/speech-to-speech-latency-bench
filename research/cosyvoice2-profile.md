# Profiling CosyVoice2

**Profiled CosyVoice2's streaming pipeline and rebuilt it to reach first audio
1.7x faster and finish 1.4x faster, after finding its LM and decoder were
contending for the Python interpreter lock rather than for the GPU.**

| | first audio | finished | LM rate |
|---|---|---|---|
| as shipped | 1701 ms | 7270 ms | 23.6 ms/token |
| rebuilt, same one GPU | 1028 ms | 6014 ms | 19.8 ms/token |
| rebuilt, decoder on a second GPU | 988 ms | ~5200 ms | 16.6 ms/token |

Same text, same model, all warm. The middle row is the like-for-like comparison
on identical hardware: 1.65x to first audio, 1.21x overall. The last row adds a
second card, which is worth nothing until the lock contention is gone and worth
a lot afterwards.

The LM's own rate is the number that explains the rest. It runs at 15.6 ms per
token with nothing else in the process, 23.6 as shipped, and 16.6 rebuilt. Most
of what looked like TTS cost was the LM waiting for the interpreter.

---

Notes from getting CosyVoice2 running and working out what streaming actually
costs, before wiring it into the harness.

One A10G on Modal, CosyVoice2-0.5B, fp16 off, cross-lingual zero-shot off the
repo's own reference wav. Input is three English sentences, which come out as
roughly 10.5 seconds of speech. Script is `modal_cosyvoice2.py`.

## How this went

I started with the sequential path, no streaming, no warmup. It finished in
around 7 seconds, which looked sensible. (Approximate: this was before the
script printed a timeline.)

Then I turned streaming on, still no warmup:

| | first chunk | total |
|---|---|---|
| streaming, no warmup | 8235 ms | 15613 ms |

Clearly wrong. Streaming is supposed to hand back audio early, and here its
*first chunk* was arriving later than the non-streaming path had finished the
entire utterance.

Cold start seemed the obvious suspect, so I added a throwaway synthesis before
the timer:

| | first chunk | total |
|---|---|---|
| streaming, no warmup | 8235 ms | 15613 ms |
| streaming, warmup | 1628 ms | 6812 ms |

Those made sense. But it still didn't explain how the un-warmed run had been so
far off, rather than just somewhat slower.

Later I reran everything, both paths, with and without warmup, just like I ran it earlier. The warm runs
came back where they had been. Un-warmed streaming did not:

| | first chunk | total |
|---|---|---|
| streaming, no warmup, first session | 8235 ms | 15613 ms |
| streaming, no warmup, later | 4122 / 4025 ms | 9249 / 8940 ms |
| streaming, warmup, first session | 1628 ms | 6812 ms |
| streaming, warmup, later | 1759 / 1714 ms | 7580 / 7110 ms |

So the warm path is reproducible to within about 5%, and the un-warmed path
halved between sessions.

I don't know what the first set of runs was doing. I ran them two or three times
and they were consistently off. The most likely candidate is network overhead I
wasn't accounting for, since the model weights live on a Modal Volume that
faults in lazily, but not proven.

## Warmup

Whatever that was, warmup on its own is worth a lot:

| streaming | first chunk | total |
|---|---|---|
| no warmup | 4122 ms | 9249 ms |
| warmup | 1759 ms | 7580 ms |

A single throwaway synthesis takes 2.4 seconds off the first chunk. The first
inference pays for CUDA kernel selection, lazy module init, and the text
frontend building its FSTs. None of that is synthesis. Un-warmed numbers are not
latency, and everything below is warm.

## CosyVoice2's own pipeline, with the decoder on a second GPU

When CosyVoice2 streams, the LM generates speech tokens in a background thread
while the main thread turns already-emitted tokens into audio. The two overlap.
Both were on the same GPU, so I assumed their kernels were being serialized
there, and moved the flow decoder and vocoder onto a second card in the same
container to find out.

It made things worse.

| | ms per token | total | first audio |
|---|---|---|---|
| non-streaming | 16.3 | 5346 ms | 5329 ms |
| non-streaming, second GPU | 15.6 | 4949 ms | 4933 ms |
| streaming | 22.4 | 7110 ms | 1714 ms |
| streaming, second GPU | 30.2 | 8796 ms | 2299 ms |

Milliseconds per token is measured from the LM's first token to its last,
divided by the tokens it emitted. The model samples rather than decoding
greedily, so the token count differs run to run and the rate is the only fair
comparison.

The same thing in absolute terms, measuring only the LM: how long it took from
its first token to its last.

| | tokens | LM first to last token |
|---|---|---|
| non-streaming, second GPU | 258 | 4003 ms |
| non-streaming | 267 | 4347 ms |
| streaming | 272 | 6061 ms |
| streaming, second GPU | 258 | 7754 ms |

The first and last rows are the comparison worth having. Both emitted **exactly
258 tokens**, on the same two cards, with the LM on `cuda:0` either way. The only
difference is whether the decoder thread was running alongside it. The LM took
4003 ms alone and 7754 ms with the decoder alive, which is 1.94x for identical
work.

In non-streaming the decoder doesn't start until the LM has finished, so the LM
has the process to itself. In streaming it does not, and it takes roughly twice
as long to emit the same tokens.

Two things fall out of that.

**Streaming costs the LM 37% of its speed**, or nearly half with the decoder on a second GPU.
16.3 ms per token with nothing else happening, 22.4 ms per token once the
decoder thread is alive. The LM is genuinely being stalled.

**A second GPU doesn't fix it, it makes it worse.** 22.4 goes to 30.2 ms per
token. If the stall were kernels queuing on a shared device, moving the decoder
to its own card would have relieved it. Instead the decoder's own passes slowed
down as well, from around 940 ms to around 1300 ms.

The likely reason is that the contention was never on the GPU. Both threads run
Python, and whichever holds the interpreter lock blocks the other regardless of
which card its kernels land on.

## Separate processes

The GIL is per-process, so the LM and the decoder were put in separate
processes, with speech tokens passed between them (`modal_cosyvoice2_split.py`). The
library's `tts()` is replaced: the LM runs on the main thread and dispatches
chunks without waiting for audio back, and a reader thread collects it.

Everything warm, 258 to 272 tokens each. The threads row reads 23.6 rather than
the 22.4 above because it was remeasured in the same file as the processes
version, so both
arms of the comparison come from identical code:

| | ms per token | total | first audio |
|---|---|---|---|
| non-streaming | 16.3 | 5346 ms | 5329 ms |
| non-streaming, second GPU | 15.6 | 4949 ms | 4933 ms |
| streaming, threads | 23.6 | 7270 ms | 1701 ms |
| streaming, threads, second GPU | 30.2 | 8796 ms | 2299 ms |
| **streaming, separate processes** | **19.8** | **6014 ms** | **1028 ms** |

**It worked, about half way.** 23.6 to 19.8 ms per token. The gap to an
unloaded LM was 8.0 ms per token and is now 4.2, so roughly 47% of the stall is
gone. Nothing changed except which process the decode runs in.

There is no non-streaming row for separate processes, because it would measure
nothing:
non-streaming already runs the decoder after the LM has finished, so there is no
contention for a process boundary to remove.

**First audio improved more than the token rate did**, 1701 to 1028 ms. Only
part of that is the processes. The library polls a shared token list every 100 ms
and our loop dispatches the moment a chunk is ready, so some of it is a polling
interval that no longer exists.

## Separate processes, and a second GPU

4.2 ms per token was still missing, and the suspect was the thing separate
processes introduce: two processes on one GPU have separate CUDA contexts, and
the driver time-slices between contexts rather than running their kernels
together. Threads at least shared a context.

So the decoder process was pinned to the second card. Unlike the threaded
attempt this adds no cross-device copies, because tensors already round-trip
through CPU to cross the process boundary.

| | ms per token | total | first audio |
|---|---|---|---|
| non-streaming | 16.3 | 5346 ms | 5329 ms |
| non-streaming, second GPU | 15.6 | 4949 ms | 4933 ms |
| streaming, threads | 23.6 | 7270 ms | 1701 ms |
| streaming, threads, second GPU | 30.2 | 8796 ms | 2299 ms |
| streaming, processes | 19.8 | 6014 ms | 1028 ms |
| **streaming, processes, second GPU** | **16.6** | **~5200 ms** | **988 ms** |

**16.6 against an unloaded 15.6.** The gap was 8.0 ms per token at the start and
is now 1.0, so roughly 87% of the stall is gone and the LM runs at close to the
rate it manages with nothing else in the process.

The order mattered. A second GPU on its own made things worse, because the GIL
was binding and moving kernels to another card could not touch it while adding
work to the decoder thread. Remove the lock first and the context switching
becomes the next constraint, at which point the second card is what removes it.
Either change alone is a loss or a half measure; together they are most of the
way.

**Streaming is close to free now.** Against non-streaming on the same two cards,
the total is 0.25 s longer instead of 1.8, and first audio arrives 3.9 s earlier.

## What streaming actually buys

All warm, same text. Non-streaming is the reference, since it has no overlap to
pay for:

| | first audio | finished | overhead vs non-streaming |
|---|---|---|---|
| non-streaming | 5.3 s | 5.3 s | — |
| streaming, shipped pipeline | 1.7 s | 7.1 s | +1.8 s |
| streaming, processes | 1.0 s | 6.0 s | +0.7 s |
| streaming, processes, second GPU | 1.0 s | 5.2 s | +0.3 s* |

**As shipped**, streaming gets audio started 3.6 seconds earlier and costs 1.8
seconds on the total. For a conversational system that is a good trade, but it
is a trade, and the overhead is the stalled LM.

**With separate processes**, it improves on both sides at once: 4.3 seconds
earlier to first audio, and the overhead falls from 1.8 to 0.7 seconds. Nothing
was made faster, the two stages just stopped taking turns.

**With a second GPU as well**, the overhead falls to 0.3 seconds, which is close
enough to free that the trade stops being a trade.

(*The first three rows are one-GPU runs, subtracted from 5346 ms. The last uses
two cards, so it is subtracted from the two-card non-streaming run at 4949 ms.
Comparing it against the one-card figure would credit the second GPU twice.)

The benchmark should quote the first of these, since it is what CosyVoice2
does out of the box. The second belongs beside it as a labelled variant.

## What to carry into the harness

1. Warm up before measuring. `S2SSystem.warmup()` already runs two throwaway
   trials, which covers it.
2. A second GPU for the TTS decoder is worth nothing on its own and hurts. It
   only pays once the decoder is in a separate process, and then it pays well.
   Budget the pair or neither.
3. Report the throughput cost of streaming alongside the latency win. As shipped
   it is 1.8 s on a 5.3 s utterance. The comparison has a real price on the
   other side, even if the optimised variant nearly erases it.
4. Pin the TTS sampling, or record the speech token count per trial. Output
   length varies about 6% run to run for identical input, which moves total time
   and RTF for reasons unrelated to latency. `check_work_constant` won't catch
   it, since it only looks at LLM tokens.
5. Steady RTF is around 0.5 to 0.65 on an A10G with nothing else on the card.
   Recheck it when the LLM shares the box, because that margin isn't large.
