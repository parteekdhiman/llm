from __future__ import annotations

from parteek_llm.rag import DocumentChunk, TfidfRetriever, chunk_text


def test_chunking_and_english_retrieval() -> None:
    chunks = [
        DocumentChunk("travel.pdf", 1, "A flight ticket can be booked online."),
        DocumentChunk("cloud.pdf", 2, "Cloud computing provides servers and storage."),
    ]
    results = TfidfRetriever(chunks).search("Where can I get cloud storage?", top_k=1)
    assert results[0].chunk.source == "cloud.pdf"
    assert results[0].score > 0


def test_hindi_retrieval() -> None:
    chunks = [
        DocumentChunk("india.pdf", 1, "भारत की राजधानी नई दिल्ली है।"),
        DocumentChunk("water.pdf", 3, "पानी गर्म होने पर भाप बनता है।"),
    ]
    results = TfidfRetriever(chunks).search("भारत की राजधानी क्या है?", top_k=1)
    assert results[0].chunk.source == "india.pdf"


def test_chunk_text_overlap() -> None:
    text = " ".join(f"word{number}" for number in range(100))
    chunks = chunk_text(text, "sample.pdf", 1, chunk_words=40, overlap_words=10)
    assert len(chunks) == 3
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]

