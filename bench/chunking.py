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

# Both require whitespace after. Without that "2.5 minutes" cuts at the decimal
# point, and "1,000" at the comma. Matching end-of-buffer would be worse still,
# because mid-stream the buffer ends wherever the last token happened to land:
# "in 2." looks like a finished sentence for as long as it takes the next token
# to arrive. A genuine trailing sentence needs no match; flush() emits it.
CLAUSE = re.compile(r"[,;:—–](?=\s)")
SENTENCE = re.compile(r"[.!?]+[\"')\]]*(?=\s)")


class ChunkPolicy:
    """Feed it token text, take back whole chunks when they are ready."""

    def __init__(
        self,
        first_chunk_words: int = 8,
        min_words: int = 6,
        max_words: int = 40,
    ):
        if first_chunk_words < 1:
            raise ValueError("first_chunk_words must be at least 1")
        self.first_chunk_words = first_chunk_words
        # A boundary below this is ignored. Punctuation arrives early and often
        # ("Jason eats 3 eggs each morning." then "He needs 90."), and cutting
        # on every one of them hands the TTS two-word fragments, which sound
        # clipped and cost a call each. Below the floor the chunk keeps growing
        # and the boundary is simply absorbed.
        self.min_words = min_words
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
        """Earliest of a usable boundary and the chunk's word cap.

        The first chunk also cuts on a clause, since it is racing to start
        audio and a clause arrives sooner than a sentence does. Later chunks
        wait for a sentence: nothing is heard while they are made.
        """
        first = self.n_emitted == 0
        cap = self.first_chunk_words if first else self.max_words
        words = _word_end_positions(self.buffer)

        candidates = []
        boundary = self._boundary_cut(words, clauses=first)
        if boundary is not None:
            candidates.append(boundary)
        if len(words) >= cap:
            candidates.append(words[cap - 1])
        return min(candidates) if candidates else None

    def _boundary_cut(self, words: list[int], clauses: bool) -> int | None:
        """First sentence (or clause) end that leaves min_words behind it."""
        if len(words) < self.min_words:
            return None
        floor = words[self.min_words - 1]
        ends = [m.end() for m in SENTENCE.finditer(self.buffer)]
        if clauses:
            ends += [m.end() for m in CLAUSE.finditer(self.buffer)]
        after = [e for e in ends if e >= floor]
        return min(after) if after else None


def _word_end_positions(text: str) -> list[int]:
    """Index just past each word that is known to be complete.

    A word only counts once whitespace or punctuation follows it. Without that
    rule the final partial token would be treated as a finished word and the
    chunk would be cut mid-word.
    """
    return [m.end() for m in re.finditer(r"\S+(?=[\s])", text)]
