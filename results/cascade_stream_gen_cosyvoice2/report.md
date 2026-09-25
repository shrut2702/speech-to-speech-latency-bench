## Pipeline, from end of speech

| config | n | time to first audio p50/p95 | end to end p50/p95 | asr final p50/p95 | llm first token p50/p95 | tts first chunk p50/p95 |
|---|---|---|---|---|---|---|
| cascade_stream_gen_cosyvoice2 | 180 | 2717 / 3715 | 12915 / 22058 | 110 / 161 | 136 / 187 | 2717 / 3715 |

## Per stage, from each stage's own start

Stage costs with everything before them subtracted out. This is the view that says which millisecond to go delete.

| config | n | asr first partial p50/p95 | asr total p50/p95 | llm ttft p50/p95 | llm 2nd token p50/p95 | llm total p50/p95 | tts first audio p50/p95 | tts total p50/p95 |
|---|---|---|---|---|---|---|---|---|
| cascade_stream_gen_cosyvoice2 | 180 | - | 108 / 158 | 24 / 29 | 7 / 8 | 440 / 736 | 2499 / 3490 | 12682 / 21803 |

## Time to first audio by length bucket

| config | short p95 | medium p95 | long p95 |
|---|---|---|---|
| cascade_stream_gen_cosyvoice2 | 3309 | 3452 | 4200 |

## Streaming health

| config | rtf p95 | max gap p95 ms | underruns | failed trials |
|---|---|---|---|---|
| cascade_stream_gen_cosyvoice2 | 0.99 | 4198 | 267 | 0 |
