"""
shared_pipeline.py

The shared post-retrieval pipeline used by all 5 RAG systems.
Ensures the only variable between systems is how they retrieve chunks,
making the comparison fair and controlled.

Pipeline steps:
  1. Rerank    — scores each chunk against the query using a CrossEncoder
                 and keeps only the most relevant ones
  2. Repack    — places the most relevant chunk last (LLMs attend more
                 strongly to text at the end of the context window)
  3. Compress  — extracts the most relevant sentences from the repacked
                 chunks to reduce noise before passing to the LLM

Requirements: pip install sentence-transformers
"""

import re
from sentence_transformers import CrossEncoder

# CrossEncoder reranker — lightweight and fast, no GPU needed
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
    """
    Score each chunk against the query using a CrossEncoder.
    Returns the top_k most relevant chunks, sorted best-first.
    """
    if not chunks:
        return []

    reranker = _get_reranker()
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for _, chunk in scored[:top_k]]
    return top_chunks


def reverse_repack(chunks: list[str]) -> list[str]:
    """
    Reverse the chunk order so the most relevant chunk appears last.
    Research shows LLMs pay stronger attention to content at the end
    of the context window (Lost in the Middle, Liu et al. 2023).
    Input is best-first; output is worst-first, best-last.
    """
    return list(reversed(chunks))


def compress(query: str, chunks: list[str], max_sentences: int = 10) -> str:
    """
    Extractive compression — splits chunks into sentences, scores each
    sentence against the query using the CrossEncoder, and returns only
    the most relevant sentences. Reduces noise before the LLM sees the context.
    """
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
    """
    Runs the full shared post-retrieval pipeline:
      1. Rerank  — keep top_k most relevant chunks
      2. Repack  — reverse order (best chunk last)
      3. Compress — extract most relevant sentences

    Args:
        query  : the customer's original question
        chunks : raw text chunks retrieved from ChromaDB
        top_k  : number of chunks to keep after reranking (default 5)

    Returns:
        A compressed context string ready to inject into the LLM prompt.
    """
    if not chunks:
        return ""

    reranked = rerank(query, chunks, top_k=top_k)
    repacked = reverse_repack(reranked)
    context = compress(query, repacked)

    return context
