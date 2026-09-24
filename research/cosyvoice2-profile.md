# Profiling CosyVoice2

**Profiled CosyVoice2's streaming pipeline and found its speech-token LM and its
flow decoder were contending for the Python interpreter lock rather than for the
GPU. Splitting them into two processes on two cards gives first audio 1.4x
faster and the whole utterance 1.4x faster, and takes the LM back to the rate it
runs at with nothing else in the process.**

Short text, about 10.4 s of speech. All warm, all one A10G unless the row says
otherwise.

| | first audio | finished | ms per token |
|---|---|---|---|
| as shipped | 1701 ms | 7270 ms | 23.6 |
| rebuilt, one GPU | 1292 ms | 6058 ms | 19.8 |
| rebuilt, two GPUs | 1173 ms | 5093 ms | 16.1 |

The LM's own rate explains the rest of the table. It runs at 15.6 ms per token
with nothing else in the process and 23.6 as shipped, so 8 ms of every token was
the LM waiting for the interpreter. The rebuild on two cards gets that back to
16.1. Most of what looked like TTS cost was not compute.

---

Notes from working out what CosyVoice2's streaming actually costs, before wiring
it into the harness. CosyVoice2-0.5B on Modal, fp16 off, cross-lingual zero shot
off the repo's own reference wav. Scripts are `modal_cosyvoice2.py` for the
shipped pipeline and `modal_cosyvoice2_split.py` for the rebuild.

## How this went

The sequential path finished in about 7 seconds, which looked sensible. Then I
turned streaming on:

| | first chunk | total |
|---|---|---|
| streaming, no warmup | 8235 ms | 15613 ms |
| streaming, warmup | 1628 ms | 6812 ms |

The un-warmed run's *first chunk* arrived later than the non-streaming path had
finished the entire utterance. Warmup explained most of it: the first inference
pays for kernel selection, lazy module init and the text frontend building its
FSTs. Everything below is warm.

What warmup did not explain was why warm streaming still cost 1.5 seconds more
in total than warm non-streaming. Streaming reorders work. It should not create
any.

## Where the time went

Timing the LM and the decoder separately, by wrapping `model.llm.inference` and
`model.token2wav`:

| | ms per token | total | first audio |
|---|---|---|---|
| non-streaming | 16.3 | 5346 ms | 5329 ms |
| non-streaming, second GPU | 15.6 | 4949 ms | 4933 ms |
| streaming | 22.4 | 7110 ms | 1714 ms |
| streaming, second GPU | 30.2 | 8796 ms | 2299 ms |

Milliseconds per token is measured from the LM's first token to its last,
divided by the tokens it emitted. The model samples rather than decoding
greedily, so the token count moves run to run and the rate is the only fair
comparison.

Two things stood out. Streaming cost the LM 6 ms per token, and adding a second
GPU made it worse rather than better.

The second one is the giveaway. If the two stages were fighting over the GPU,
giving them a card each would have helped. It did the opposite, which means they
were fighting over something a second card cannot divide: the interpreter lock.
CosyVoice runs the LM on a background thread while the main thread decodes, and
Python lets only one thread run Python code at a time. Moving the decoder to
another card just added device transfers to a thread that was already waiting.

## The rebuild

Two processes rather than two threads. Process A runs the LM and dispatches
token chunks; process B runs the flow decoder and the vocoder and sends audio
back. Separate interpreters, so no lock to share.

Short text, one GPU:

| | ms per token | total | first audio |
|---|---|---|---|
| as shipped | 23.6 | 7270 ms | 1701 ms |
| rebuilt | 19.8 | 6058 ms | 1292 ms |

About half the stall recovered. The gap to an unloaded LM was 8.0 ms per token
and is now 4.2. What was left is the thing two processes on one card still
share, which is the card: CUDA has to context switch between them, and that is
real work a thread swap does not do.

Giving the decoder its own card removes that too:

| | ms per token | total | first audio |
|---|---|---|---|
| as shipped | 23.6 | 7270 ms | 1701 ms |
| rebuilt, two GPUs | 16.1 | 5093 ms | 1173 ms |

16.1 against an unloaded 15.6. The contention is gone.

## Does it hold on longer text

Everything above is about 10.4 s of speech. Repeated on a paragraph of roughly
30 s, which the library splits into two pieces:

| | first audio | finished | audio | wall clock per second of audio | ms per token |
|---|---|---|---|---|---|
| as shipped | 1807 ms | 20874 ms | 29.7 s | 0.703 | 23.0 then 20.1 |
| rebuilt, one GPU | 1285 ms | 18727 ms | 30.0 s | 0.624 | 22.1 then 19.5 |
| rebuilt, two GPUs | 1329 ms | 15670 ms | 30.0 s | 0.522 | 17.2 then 16.9 |

Sampling gives different token counts per run, so the totals are normalized by
the audio actually produced.

The two-GPU rebuild holds: 16.1 ms per token on short text, 17.2 and 16.9 here.
The one-GPU rebuild degrades, from 19.8 to 22.1, because each decode re-runs the
whole sequence so far and later passes are much longer, which means longer
stretches where one process owns the card.

| | first audio | total, normalized |
|---|---|---|
| one GPU, short | 1.32x | 1.18x |
| one GPU, long | 1.41x | 1.13x |
| two GPUs, short | 1.45x | 1.41x |
| two GPUs, long | 1.36x | 1.35x |

## Three things the rebuild had to copy exactly

The first version of the rebuild looked better than it was, in three separate
ways, all of which came from simplifying the library's chunk schedule instead of
reproducing it.

**The first chunk is padded.** The flow decoder blocks the sequence into
`token_hop_len` pieces counted from the first *prompt* token, and a prompt
rarely ends on a block boundary, so the library makes the first generated chunk
absorb the remainder. Skipping that gave a first chunk of 21 tokens against the
library's 34, which is a lower time to first audio for no reason at all.

**The hop cap is 100, not 200.** The attribute is `token_max_hop_len`, set to
four times `token_hop_len`. I had guessed at `max_token_hop_len`, which does not
exist, so a `getattr` default of 200 quietly took over and chunks grew to 8
seconds. Bigger chunks mean fewer decoder passes, which flattered the total.

**The library splits long text into pieces.** `inference_cross_lingual` loops
over `text_normalize(split=True)`, which cuts English into 60 to 80 token
pieces, and runs each one through the LM and the decoder in turn. The rebuild
was feeding whole paragraphs as a single sequence, so it never paid the gap
between pieces and was generating longer sequences than the model normally sees.

Each of those made the rebuild look faster than it is. The numbers above are
after all three were fixed.

## Two bugs worth reporting upstream

Both are visible in the shipped pipeline's own output and neither needs extra
hardware to fix.

**`token_hop_len` is never reset.** `CosyVoice2Model.tts` grows it in place and
leaves it there, so the next utterance starts wherever the last one finished. In
the long-paragraph run, piece two's first chunk is 109 tokens instead of 34,
because piece one left the hop at 100. Time to first audio is four times worse
on every utterance after the first.

**Consecutive pieces are strictly sequential.** Piece two's LM does not start
until piece one's decoder has finished. That is 3.5 seconds of idle LM in the
shipped long-paragraph run. The two could overlap, since they touch different
state.

## What streaming actually buys

All warm, same short text. Non-streaming is the reference, since it has no
overlap to pay for:

| | first audio | finished | overhead vs non-streaming |
|---|---|---|---|
| non-streaming | 5.3 s | 5.3 s | |
| streaming, shipped pipeline | 1.7 s | 7.1 s | +1.8 s |
| streaming, rebuilt, one GPU | 1.3 s | 6.1 s | +0.7 s |
| streaming, rebuilt, two GPUs | 1.2 s | 5.1 s | +0.1 s* |

As shipped, streaming gets audio started 3.6 seconds earlier and costs 1.8
seconds on the total. For a conversational system that is a good trade, but it
is a trade, and the overhead is the stalled LM.

With two processes on one card it improves on both sides at once: 4.0 seconds
earlier to first audio, and the overhead falls from 1.8 to 0.7 seconds. Nothing
was made faster. The two stages just stopped taking turns.

With a second card as well the overhead is 0.1 seconds, which is close enough to
free that the trade stops being a trade.

(*The first three rows are one-GPU runs, measured against the one-GPU
non-streaming time of 5346 ms. The last uses two cards, so it is measured
against the two-card non-streaming run at 4949 ms. Comparing it against the
one-card figure would credit the second GPU twice.)
