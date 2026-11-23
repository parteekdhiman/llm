from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


def chunk_text(
    text: str,
    source: str,
    page: int,
    *,
    chunk_words: int = 180,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    if chunk_words < 20:
        raise ValueError("chunk_words must be at least 20")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words")
    cleaned = WHITESPACE.sub(" ", text).strip()
    words = cleaned.split()
    if not words:
        return []
    step = chunk_words - overlap_words
    chunks: list[DocumentChunk] = []
    for start in range(0, len(words), step):
        selected = words[start : start + chunk_words]
        if len(selected) < 12 and chunks:
            break
        chunks.append(DocumentChunk(source=source, page=page, text=" ".join(selected)))
        if start + chunk_words >= len(words):
            break
    return chunks


def extract_pdf_chunks(
    pdf_paths: list[str | Path],
    *,
    chunk_words: int = 180,
    overlap_words: int = 40,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for raw_path in pdf_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        try:
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                chunks.extend(
                    chunk_text(
                        text,
                        path.name,
                        page_number,
                        chunk_words=chunk_words,
                        overlap_words=overlap_words,
                    )
                )
        except Exception as exc:
            raise RuntimeError(f"Could not read {path}: {exc}") from exc
    if not chunks:
        raise ValueError(
            "No extractable PDF text was found. Scanned PDFs require OCR before using this tool."
        )
    return chunks


def discover_pdfs(directory: str | Path) -> list[Path]:
    path = Path(directory)
    if not path.is_dir():
        raise NotADirectoryError(f"PDF directory not found: {path}")
    pdfs = sorted(path.rglob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found under {path}")
    return pdfs


class TfidfRetriever:
    """Character n-gram retrieval that works across English and Devanagari."""

    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("At least one document chunk is required")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
            max_features=150_000,
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    def search(self, query: str, top_k: int = 4) -> list[SearchResult]:
        if not query.strip() or top_k < 1:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        best_indices = np.argsort(scores)[::-1][: min(top_k, len(self.chunks))]
        return [
            SearchResult(self.chunks[int(index)], float(scores[int(index)]))
            for index in best_indices
        ]


def format_retrieved_context(results: list[SearchResult]) -> str:
    passages = []
    for number, result in enumerate(results, start=1):
        passages.append(
            f"[{number}] Source: {result.chunk.source}, page {result.chunk.page}\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(passages)

