"""
hybrid_rag.py

Hybrid RAG pipeline.
Combines dense vector search (ChromaDB) with sparse keyword search (BM25)
and merges results using Reciprocal Rank Fusion (RRF).

Dense retrieval is good at finding semantically similar content.
Sparse retrieval is good at finding exact keyword matches (e.g. product SKUs,
policy terms). Combining both captures what either alone would miss.
"""

import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
from pipelines.shared_pipeline import shared_pipeline
from pipelines.llm import generate_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 10      # chunks retrieved from each method before fusion
RRF_K = 60      # RRF constant — standard value from the original paper


def reciprocal_rank_fusion(dense_chunks: list[str], sparse_chunks: list[str]) -> list[str]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score for a document: sum(1 / (k + rank)) across all lists.
    Higher score = more relevant across both retrieval methods.
    """
    scores = {}
    for rank, chunk in enumerate(dense_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (RRF_K + rank + 1)
    for rank, chunk in enumerate(sparse_chunks):
        scores[chunk] = scores.get(chunk, 0) + 1 / (RRF_K + rank + 1)

    sorted_chunks = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return sorted_chunks


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="knowledge_base",
        persist_directory=os.path.join(CHROMA_DIR, "knowledge_base"),
        embedding_function=embeddings,
    )


def build_bm25_index(vectorstore: Chroma):
    """Build a BM25 index from all chunks stored in ChromaDB."""
    all_docs = vectorstore.get()
    corpus = all_docs["documents"]
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised)
    return bm25, corpus


def run(query: str, vectorstore=None, bm25=None, corpus=None) -> dict:
    """
    Run the Hybrid RAG pipeline.

    Args:
        query       : the customer's question
        vectorstore : optional preloaded ChromaDB instance
        bm25        : optional preloaded BM25 index
        corpus      : optional list of all corpus documents (parallel to BM25 index)

    Returns:
        dict with keys: response, context, retrieved_chunks, latency_seconds
    """
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    if bm25 is None or corpus is None:
        bm25, corpus = build_bm25_index(vectorstore)

    # Step 1 — Dense retrieval via ChromaDB
    dense_results = vectorstore.similarity_search(query, k=TOP_K)
    dense_chunks = [doc.page_content for doc in dense_results]

    # Step 2 — Sparse retrieval via BM25
    tokenised_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    sparse_chunks = [corpus[i] for i in top_bm25_indices]

    # Step 3 — Merge with Reciprocal Rank Fusion
    fused_chunks = reciprocal_rank_fusion(dense_chunks, sparse_chunks)

    # Step 4 — Shared pipeline (rerank → repack → compress)
    context = shared_pipeline(query, fused_chunks)

    # Step 5 — Generate response
    response = generate_response(query=query, context=context)

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "hybrid_rag",
        "query": query,
        "response": response,
        "context": context,
        "retrieved_chunks": fused_chunks,
        "dense_chunks": dense_chunks,
        "sparse_chunks": sparse_chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "Do you offer free delivery?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
