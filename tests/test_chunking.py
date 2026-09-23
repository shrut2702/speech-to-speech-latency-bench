"""The chunking policy decides time to first audio, so it gets tested."""

from bench.chunking import ChunkPolicy


def feed_words(policy: ChunkPolicy, text: str) -> list[str]:
    """Streams text the way an LLM does, one word at a time."""
    out = []
    for word in text.split():
        out += policy.feed(word + " ")
    return out + policy.flush()


def test_first_chunk_is_short_and_the_rest_are_sentences():
    chunks = feed_words(
        ChunkPolicy(first_chunk_words=4),
        "The capital of France is Paris. It has been the capital since 508.",
    )
    assert chunks[0] == "The capital of France"
    assert chunks[1] == "is Paris. It has been the capital since 508."


def test_first_chunk_cuts_at_a_clause_when_one_arrives_sooner():
    # Waiting for the tenth word would delay audio for no reason when the
    # phrase already ended.
    chunks = feed_words(
        ChunkPolicy(first_chunk_words=10, min_words=3),
        "The capital of France, which is Paris, is a large city.",
    )
    assert chunks[0] == "The capital of France,"


def test_a_decimal_point_is_not_a_sentence():
    # The buffer ends wherever the last token landed, so "2." looked like a
    # finished sentence until the next token arrived.
    chunks = feed_words(
        ChunkPolicy(first_chunk_words=20),
        "It takes about 2.5 minutes, or 1,000 seconds at worst.",
    )
    assert chunks[0].startswith("It takes about 2.5 minutes")


def test_a_sentence_that_ends_too_soon_is_absorbed():
    # "He needs 90." on its own is a two-word chunk: clipped to listen to and
    # a whole TTS call for nothing.
    chunks = feed_words(
        ChunkPolicy(first_chunk_words=4, min_words=6),
        "Jason eats 3 eggs each morning. He needs 90. That is a lot of eggs.",
    )
    assert chunks[0] == "Jason eats 3 eggs"
    assert all(len(c.split()) >= 6 for c in chunks[1:-1])


def test_only_the_first_chunk_is_cut_short():
    # The whole point: later chunks hide inside the playback of earlier ones,
    # so they stay sentence-sized rather than being cut to the same length.
    chunks = feed_words(
        ChunkPolicy(first_chunk_words=2),
        "Yes indeed. The second sentence here is considerably longer than two words.",
    )
    assert chunks[0] == "Yes indeed."
    assert len(chunks[1].split()) > 2


def test_never_cuts_inside_a_word():
    # Tokens arrive as sub-word pieces, so a word only counts once something
    # follows it.
    policy = ChunkPolicy(first_chunk_words=3)
    emitted = []
    for piece in ["The ", "cap", "ital ", "of ", "Fra", "nce ", "is ", "Paris."]:
        emitted += policy.feed(piece)
    emitted += policy.flush()
    assert emitted[0] == "The capital of"
    assert "".join(emitted).replace(" ", "") == "ThecapitalofFranceisParis."


def test_a_model_that_never_punctuates_still_streams():
    # Without the word cap this would buffer the whole response and collapse
    # stream_gen back into batch.
    policy = ChunkPolicy(first_chunk_words=3, max_words=5)
    chunks = feed_words(policy, " ".join(f"w{i}" for i in range(20)))
    assert chunks[0] == "w0 w1 w2"
    assert all(len(c.split()) <= 5 for c in chunks[1:])
    assert len(chunks) > 2


def test_flush_emits_a_trailing_fragment():
    policy = ChunkPolicy(first_chunk_words=2)
    out = policy.feed("Hello there ") + policy.feed("friend")
    assert policy.flush() == ["friend"]
    assert out == ["Hello there"]


def test_empty_stream_emits_nothing():
    policy = ChunkPolicy()
    assert policy.feed("") == []
    assert policy.flush() == []
