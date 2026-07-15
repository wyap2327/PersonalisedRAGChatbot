import os
import time
import ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pipelines.shared_pipeline import shared_pipeline
from pipelines.llm import generate_response, MODEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5 # chunks retrieved per query variant
NUM_VARIANTS = 3 # number of alternative queries to generate
PERSONA_TOP_K = 3  # persona chunks retrieved per query variant when a customer is identified


def generate_query_variants(query: str) -> list[str]:
    prompt = f"""Generate {NUM_VARIANTS} different ways to ask the following customer support question.
    Each rephrasing should capture the same intent but use different wording.
    Return only the rephrased questions, one per line with no numbering or extra text.
    Original question: {query}"""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
    return variants[:NUM_VARIANTS]


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


def run(query: str, vectorstore=None, customer_id: str = None, persona_vectorstore=None) -> dict:
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 1 — Generate alternative query phrasings
    variants = generate_query_variants(query)
    all_queries = [query] + variants  # include original query

    # Step 2 — Retrieve chunks for each query variant
    seen = set()
    retrieved_chunks = []
    for q in all_queries:
        results = vectorstore.similarity_search(q, k=TOP_K)
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                retrieved_chunks.append(doc.page_content)

    # Step 2b — If a customer is identified, retrieve persona chunks for each query variant too
    persona_chunks = []
    if customer_id:
        if persona_vectorstore is None:
            persona_vectorstore = load_persona_vectorstore()
        persona_seen = set()
        for q in all_queries:
            persona_results = persona_vectorstore.similarity_search(
                q, k=PERSONA_TOP_K, filter={"customer_id": customer_id}
            )
            for doc in persona_results:
                if doc.page_content not in persona_seen:
                    persona_seen.add(doc.page_content)
                    persona_chunks.append(doc.page_content)

    # Step 3 — Shared pipeline (rerank → repack → compress) over KB + persona chunks combined
    context = shared_pipeline(query, retrieved_chunks + persona_chunks)

    # Step 4 — Generate response
    response = generate_response(query=query, context=context)

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "multiquery_rag",
        "query": query,
        "query_variants": variants,
        "response": response,
        "context": context,
        "retrieved_chunks": retrieved_chunks,
        "persona_chunks": persona_chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "What is my loyalty points balance and how do I redeem them?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Query variants generated: {result['query_variants']}\n")
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
