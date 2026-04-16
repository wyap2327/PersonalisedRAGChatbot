"""
contextual_rag.py

Contextual (Personalised) RAG pipeline.
Retrieves the customer's profile from ChromaDB using their customer_id,
enriches the query with their membership tier, order history, and preferences,
then retrieves relevant knowledge base chunks using the enriched query.

This is the key hypothesis of the dissertation — that personalising retrieval
using customer profile data produces more relevant and satisfying responses
compared to non-personalised RAG strategies.
"""

import os
import json
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pipelines.shared_pipeline import shared_pipeline
from pipelines.llm import generate_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
CUSTOMERS_JSON = os.path.join(BASE_DIR, "Customer & Persona", "customer_data", "customers.json")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 10


def load_customer(customer_id: str) -> dict | None:
    """Look up a customer record by customer_id from customers.json."""
    with open(CUSTOMERS_JSON, "r", encoding="utf-8") as f:
        customers = json.load(f)
    for customer in customers:
        if customer["customer_id"] == customer_id:
            return customer
    return None


def build_enriched_query(query: str, customer: dict) -> str:
    """
    Enrich the original query with customer profile context.
    The enriched query is used for ChromaDB retrieval only —
    the customer's name and tier are also injected into the LLM prompt.
    """
    prefs = customer["preferences"]
    account = customer["account"]
    recent_orders = customer["order_history"][:2]
    recent_products = [
        item["product_name"]
        for order in recent_orders
        for item in order["items"]
    ]

    enriched = (
        f"Customer profile: {account['membership_tier']} tier member. "
        f"Sport interests: {', '.join(prefs['sport_interests'])}. "
        f"Preferred brands: {', '.join(prefs['preferred_brands'])}. "
        f"Recent purchases: {', '.join(recent_products[:3])}. "
        f"Query: {query}"
    )
    return enriched


def build_customer_context(customer: dict) -> str:
    """Build a plain-text customer summary to inject into the LLM prompt."""
    p = customer["personal_details"]
    a = customer["account"]
    prefs = customer["preferences"]
    recent_orders = customer["order_history"][:2]
    recent_products = [
        item["product_name"]
        for order in recent_orders
        for item in order["items"]
    ]

    lines = [
        f"Customer: {p['full_name']}",
        f"Membership tier: {a['membership_tier']}",
        f"Loyalty points: {a['loyalty_points_balance']} points",
        f"Sport interests: {', '.join(prefs['sport_interests'])}",
        f"Preferred brands: {', '.join(prefs['preferred_brands'])}",
        f"Shoe size (UK): {prefs['shoe_size_uk']}",
        f"Clothing size: {prefs['clothing_size']}",
        f"Recent purchases: {', '.join(recent_products[:3]) if recent_products else 'None'}",
    ]

    if customer["past_interactions"]:
        last = customer["past_interactions"][-1]
        lines.append(f"Last contact ({last['date']}): {last['detail']}")

    return "\n".join(lines)


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


def run(query: str, customer_id: str, vectorstore=None) -> dict:
    """
    Run the Contextual RAG pipeline.

    Args:
        query       : the customer's question
        customer_id : e.g. 'CUST-001' — used to load their profile
        vectorstore : optional preloaded ChromaDB instance

    Returns:
        dict with keys: response, context, customer_profile, retrieved_chunks, latency_seconds
    """
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    # Step 1 — Load customer profile
    customer = load_customer(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found."}

    # Step 2 — Enrich query with customer profile for retrieval
    enriched_query = build_enriched_query(query, customer)

    # Step 3 — Retrieve top-k chunks using the enriched query
    results = vectorstore.similarity_search(enriched_query, k=TOP_K)
    retrieved_chunks = [doc.page_content for doc in results]

    # Step 4 — Shared pipeline (rerank → repack → compress)
    context = shared_pipeline(query, retrieved_chunks)

    # Step 5 — Build customer context string for LLM prompt
    customer_context = build_customer_context(customer)
    full_context = f"--- Customer Profile ---\n{customer_context}\n\n--- Knowledge Base ---\n{context}"

    # Step 6 — Generate personalised response
    customer_name = customer["personal_details"]["first_name"]
    response = generate_response(
        query=query,
        context=full_context,
        customer_name=customer_name,
    )

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "contextual_rag",
        "query": query,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "enriched_query": enriched_query,
        "response": response,
        "context": full_context,
        "retrieved_chunks": retrieved_chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "What is my loyalty points balance and how do I redeem them?"
    customer_id = "CUST-001"
    print(f"Query: {query}")
    print(f"Customer: {customer_id}\n")
    result = run(query, customer_id)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
