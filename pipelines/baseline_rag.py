
import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pipelines.shared_pipeline import shared_pipeline
from pipelines.llm import generate_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 10  # chunks retrieved before reranking narrows to 5


def load_vectorstore(): # Connects to the ChromaDB collection
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


def run(query: str, vectorstore=None) -> dict:
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 1 — Retrieve top-k chunks from ChromaDB
    results = vectorstore.similarity_search(query, k=TOP_K)
    retrieved_chunks = [doc.page_content for doc in results]

    # Step 2 — Shared pipeline (rerank → repack → compress)
    context = shared_pipeline(query, retrieved_chunks)

    # Step 3 — Generate response
    response = generate_response(query=query, context=context)

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "baseline_rag",
        "query": query,
        "response": response,
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "What is my loyalty points balance and how do I redeem them?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
