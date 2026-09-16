# Profiling CosyVoice2

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

## Running the LM and decoder in parallel

When CosyVoice2 streams, the LM generates speech tokens in a background thread
while the main thread turns already-emitted tokens into audio. The two overlap.
Both were on the same GPU, so I assumed their kernels were being serialized
there, and moved the flow decoder and vocoder onto a second card in the same
container to find out.

It made things worse.

| | ms per token | total | first audio |
|---|---|---|---|
| non-streaming | 16.3 | 5346 ms | 5329 ms |
| non-streaming, split | 15.6 | 4949 ms | 4933 ms |
| streaming | 22.4 | 7110 ms | 1714 ms |
| streaming, split | 30.2 | 8796 ms | 2299 ms |

Milliseconds per token is measured from the LM's first token to its last,
divided by the tokens it emitted. The model samples rather than decoding
greedily, so the token count differs run to run and the rate is the only fair
comparison.

The same thing in absolute terms, measuring only the LM: how long it took from
its first token to its last.

| | tokens | LM first to last token |
|---|---|---|
| non-streaming, split | 258 | 4003 ms |
| non-streaming | 267 | 4347 ms |
| streaming | 272 | 6061 ms |
| streaming, split | 258 | 7754 ms |

The first and last rows are the comparison worth having. Both emitted **exactly
258 tokens**, on the same two cards, with the LM on `cuda:0` either way. The only
difference is whether the decoder thread was running alongside it. The LM took
4003 ms alone and 7754 ms with the decoder alive, which is 1.94x for identical
work.

In non-streaming the decoder doesn't start until the LM has finished, so the LM
has the process to itself. In streaming it does not, and it takes roughly twice
as long to emit the same tokens.

Two things fall out of that.

**Streaming costs the LM 37% of its speed**, or nearly half in the split case.
16.3 ms per token with nothing else happening, 22.4 ms per token once the
decoder thread is alive. The LM is genuinely being stalled.

**A second GPU doesn't fix it, it makes it worse.** 22.4 goes to 30.2 ms per
token. If the stall were kernels queuing on a shared device, moving the decoder
to its own card would have relieved it. Instead the decoder's own passes slowed
down as well, from around 940 ms to around 1300 ms.

The likely reason is that the contention was never on the GPU. Both threads run
Python, and whichever holds the interpreter lock blocks the other regardless of
which card its kernels land on. Splitting devices added cross-device copies and
synchronisation to the decoder thread, so it holds the lock longer and starves
the LM more.

## Headroom, not taken

The stall is fixable. The GIL is per-process, so running the LM and the decoder
as two processes with tokens passed between them would give real parallelism and
should recover most of that 1.94x. 

## What streaming actually buys

Both warm, same text:

- streaming: first audio at 1.7 s, finished at 7.1 s
- non-streaming: first audio at 5.3 s, finished at 5.3 s

Streaming gets audio started 3.6 seconds earlier and takes 1.8 seconds longer
overall. For a conversational system that is a good trade, but it is a trade,
not a free win.

## What to carry into the harness

1. Warm up before measuring. `S2SSystem.warmup()` already runs two throwaway
   trials, which covers it.
2. Don't budget a GPU for the TTS decoder. It measurably hurts.
3. Report the throughput cost of streaming alongside the latency win. The
   streaming-versus-batch comparison has a real price on the other side.
4. Pin the TTS sampling, or record the speech token count per trial. Output
   length varies about 6% run to run for identical input, which moves total time
   and RTF for reasons unrelated to latency. `check_work_constant` won't catch
   it, since it only looks at LLM tokens.
5. Steady RTF is around 0.5 to 0.65 on an A10G with nothing else on the card.
   Recheck it when the LLM shares the box, because that margin isn't large.
