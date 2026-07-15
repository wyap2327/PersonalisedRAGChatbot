"""
Dense retrieval - semantic similarity search
Sparse retrieval - keyword-matching search (BM25)
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
PERSONA_TOP_K = 3  # persona chunks retrieved from each method when a customer is identified

# Takes the two result lists and merges them into one ranked list. 
# Each chunk gets a score - a chunk appearing near the top of both gets the highest combined score.
def reciprocal_rank_fusion(dense_chunks: list[str], sparse_chunks: list[str]) -> list[str]:
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
    all_docs = vectorstore.get()
    corpus = all_docs["documents"]
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised) # keyword search index
    return bm25, corpus


def load_persona_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name="customer_personas",
        persist_directory=os.path.join(CHROMA_DIR, "customer_personas"),
        embedding_function=embeddings,
    )


def build_persona_bm25_index(persona_vectorstore: Chroma, customer_id: str):
    # Only this customer's own chunks — BM25 has no built-in metadata filter, so the corpus is pre-filtered
    docs = persona_vectorstore.get(where={"customer_id": customer_id})
    corpus = docs["documents"]
    if not corpus:
        return None, []
    tokenised = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenised)
    return bm25, corpus


def run(query: str, vectorstore=None, bm25=None, corpus=None, customer_id: str = None, persona_vectorstore=None) -> dict:
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    if bm25 is None or corpus is None:
        bm25, corpus = build_bm25_index(vectorstore)

    # Step 1 — semantic similarity search
    dense_results = vectorstore.similarity_search(query, k=TOP_K)
    dense_chunks = [doc.page_content for doc in dense_results]

    # Step 2 — keyword-matching search via BM25
    tokenised_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenised_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    sparse_chunks = [corpus[i] for i in top_bm25_indices]

    # Step 2b — If a customer is identified, run the same dense+sparse fusion against their persona data
    persona_chunks = []
    if customer_id:
        if persona_vectorstore is None:
            persona_vectorstore = load_persona_vectorstore()

        persona_dense_results = persona_vectorstore.similarity_search(
            query, k=PERSONA_TOP_K, filter={"customer_id": customer_id}
        )
        persona_dense_chunks = [doc.page_content for doc in persona_dense_results]

        persona_bm25, persona_corpus = build_persona_bm25_index(persona_vectorstore, customer_id)
        persona_sparse_chunks = []
        if persona_bm25 is not None:
            persona_bm25_scores = persona_bm25.get_scores(tokenised_query)
            top_persona_indices = sorted(
                range(len(persona_bm25_scores)), key=lambda i: persona_bm25_scores[i], reverse=True
            )[:PERSONA_TOP_K]
            persona_sparse_chunks = [persona_corpus[i] for i in top_persona_indices]

        persona_chunks = reciprocal_rank_fusion(persona_dense_chunks, persona_sparse_chunks)

    # Step 3 — Merge KB results with Reciprocal Rank Fusion
    fused_chunks = reciprocal_rank_fusion(dense_chunks, sparse_chunks)

    # Step 4 — Shared pipeline (rerank → repack → compress) over KB + persona results combined
    context = shared_pipeline(query, fused_chunks + persona_chunks)

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
        "persona_chunks": persona_chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "Do you offer free delivery?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
