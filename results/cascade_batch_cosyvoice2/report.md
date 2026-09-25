## Pipeline, from end of speech

| config | n | time to first audio p50/p95 | end to end p50/p95 | asr final p50/p95 | llm first token p50/p95 | tts first chunk p50/p95 |
|---|---|---|---|---|---|---|
| cascade_batch_cosyvoice2 | 180 | 11892 / 17454 | 11893 / 17456 | 129 / 167 | 157 / 198 | 11891 / 17452 |

## Per stage, from each stage's own start

Stage costs with everything before them subtracted out. This is the view that says which millisecond to go delete.

| config | n | asr first partial p50/p95 | asr total p50/p95 | llm ttft p50/p95 | llm 2nd token p50/p95 | llm total p50/p95 | tts first audio p50/p95 | tts total p50/p95 |
|---|---|---|---|---|---|---|---|---|
| cascade_batch_cosyvoice2 | 180 | - | 127 / 164 | 27 / 34 | 7 / 8 | 443 / 739 | 11226 / 16758 | 11226 / 16758 |

## Time to first audio by length bucket

| config | short p95 | medium p95 | long p95 |
|---|---|---|---|
| cascade_batch_cosyvoice2 | 13758 | 16423 | 20302 |

## Streaming health

| config | rtf p95 | max gap p95 ms | underruns | failed trials |
|---|---|---|---|---|
| cascade_batch_cosyvoice2 | 1.03 | nan | 0 | 0 |
