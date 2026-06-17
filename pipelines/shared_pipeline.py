"""
Requirements: pip install sentence-transformers
"""

import re
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def _get_reranker():
    """Load the reranker once and reuse it."""
    global _reranker
    if _reranker is None:
        print(f"  Loading reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]
    return top_chunks


def reverse_repack(chunks: list[str]) -> list[str]:
    return list(reversed(chunks))


def compress(query: str, chunks: list[str], max_sentences: int = 10) -> str:
    if not chunks:
        return ""

    # Split all chunks into individual sentences
    full_text = " ".join(chunks)
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return full_text

    # Score each sentence against the query
    reranker = _get_reranker()
    pairs = [(query, sentence) for sentence in sentences]
    scores = reranker.predict(pairs)

    # Select top sentences and return them in their original order
    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted([i for i, _ in scored[:max_sentences]])
    selected = [sentences[i] for i in top_indices]

    return " ".join(selected)


def shared_pipeline(query: str, chunks: list[str], top_k: int = 5) -> str:
    if not chunks:
        return ""

    reranked = rerank(query, chunks, top_k=top_k)
    repacked = reverse_repack(reranked)
    context = compress(query, repacked)

    return context
