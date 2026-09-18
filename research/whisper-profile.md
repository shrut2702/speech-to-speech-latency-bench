# Profiling Whisper

One A10G on Modal, `large-v3-turbo`, float16, greedy. Warm. Both paths load the
same weights, so the only difference is policy. Script is `modal_whisper.py`.

Streaming re-runs inference every second and commits what two consecutive runs
agree on. It is fed at wall clock pace, because handing it the whole array at
once would make it batch with extra steps.

The last two clips are stitched from the real speech. The set tops out under 19
seconds and Whisper pads everything to a 30 second window, so nothing in it can
show what happens once an utterance needs a second window.

## Latency from the endpoint

What each path still owes when the user stops talking. Everything before that is
free as far as the wait goes.

| speech | batch | streaming | streaming gpu busy | passes |
|---|---|---|---|---|
| 2.26 s | 137 ms | 165 ms | 484 ms | 3 |
| 6.92 s | 164 ms | 201 ms | 1270 ms | 7 |
| 9.36 s | 175 ms | 214 ms | 1892 ms | 10 |
| 40.0 s | 500 ms | 352 ms | 13892 ms | 41 |
| 80.0 s | 877 ms | 213 ms | 26932 ms | 81 |

## Streaming loses on every clip we have

On all three real clips, streaming is slower at the endpoint than batch. Not by
much, 30 to 40 ms, but the wrong way round from what the project assumed.

The reason is that batch is already cheap here. Everything under 30 seconds fits
in one encoder window, so a whole transcription costs 137 to 175 ms no matter
how long the clip is. 

Streaming does win past 30 seconds, where batch starts paying for extra windows
and streaming does not. Its endpoint cost stays roughly flat because the buffer
gets trimmed at each commit, so the finalize is always about one chunk's worth
of audio whether the utterance ran 40 seconds or 80.

The crossover sits somewhere near 30 seconds. Every clip in the benchmark is
under 19.

## The GPU cost is the real problem

Streaming runs a full inference pass every second, so a 9 second clip costs 10
passes instead of 1.

| speech | batch gpu time | streaming gpu time | ratio |
|---|---|---|---|
| 2.26 s | 137 ms | 484 ms | 3.5x |
| 9.36 s | 175 ms | 1892 ms | 11x |
| 80.0 s | 877 ms | 26932 ms | 31x |

