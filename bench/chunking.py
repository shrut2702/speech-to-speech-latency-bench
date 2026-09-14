"""Cuts an LLM token stream into TTS chunks.

The policy is deliberately uneven: a small first chunk, sentence-sized chunks
after it.

Only the first chunk's synthesis time is ever heard. Once audio is playing you
have the whole duration of chunk N to produce chunk N+1, so a later chunk being
slower costs nothing as long as synthesis stays faster than playback. Making
every chunk small would trade prosody away for a win that only the first chunk
can actually collect.

Per-token synthesis is the failure mode this avoids. A TTS needs a phrase to
place stress and intonation, so word-by-word output is flat and disjointed, and
the fixed per-call overhead makes it slower overall as well.

`first_chunk_words` is a knob worth sweeping rather than guessing: it trades
time to first audio directly against how natural chunk one sounds.
"""

from __future__ import annotations

import re

# A clause boundary is good enough to start speaking on, and usually arrives
# sooner than a sentence does.
CLAUSE = re.compile(r"[,;:—–]")
SENTENCE = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)")


class ChunkPolicy:
    """Feed it token text, take back whole chunks when they are ready."""

    def __init__(self, first_chunk_words: int = 4, max_words: int = 40):
        if first_chunk_words < 1:
            raise ValueError("first_chunk_words must be at least 1")
        self.first_chunk_words = first_chunk_words
        # Safety valve. A model that never punctuates would otherwise buffer
        # its whole response and collapse stream_gen back into batch.
        self.max_words = max_words
        self.buffer = ""
        self.n_emitted = 0

    def feed(self, token: str) -> list[str]:
        """Adds token text, returns any chunks that are now complete."""
        self.buffer += token
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk, self.buffer = self.buffer[:cut].strip(), self.buffer[cut:]
            if chunk:
                out.append(chunk)
                self.n_emitted += 1
        return out

    def flush(self) -> list[str]:
        """Emits whatever is left once the LLM is done."""
        rest = self.buffer.strip()
        self.buffer = ""
        if not rest:
            return []
        self.n_emitted += 1
        return [rest]

    # ---- internals -------------------------------------------------------

    def _find_cut(self) -> int | None:
        if self.n_emitted == 0:
            return self._first_cut()
        return self._sentence_cut() or self._word_cap_cut()

    def _first_cut(self) -> int | None:
        """Earliest of: a clause boundary, a sentence end, or N whole words.

        Only cuts where a word has actually finished, since tokens arrive as
        sub-word pieces and splitting inside one would hand the TTS a fragment.
        """
        sent = SENTENCE.search(self.buffer)
        clause = CLAUSE.search(self.buffer)
        candidates = [m.end() for m in (sent, clause) if m is not None]

        words = _word_end_positions(self.buffer)
        if len(words) >= self.first_chunk_words:
            candidates.append(words[self.first_chunk_words - 1])

        return min(candidates) if candidates else None

    def _sentence_cut(self) -> int | None:
        m = SENTENCE.search(self.buffer)
        return m.end() if m else None

    def _word_cap_cut(self) -> int | None:
        words = _word_end_positions(self.buffer)
        if len(words) >= self.max_words:
            return words[self.max_words - 1]
        return None


def _word_end_positions(text: str) -> list[int]:
    """Index just past each word that is known to be complete.

    A word only counts once whitespace or punctuation follows it. Without that
    rule the final partial token would be treated as a finished word and the
    chunk would be cut mid-word.
    """
    return [m.end() for m in re.finditer(r"\S+(?=[\s])", text)]
