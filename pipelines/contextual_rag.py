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
PERSONA_TOP_K = 3


def load_customer(customer_id: str) -> dict | None:
    with open(CUSTOMERS_JSON, "r", encoding="utf-8") as f:
        customers = json.load(f)
    for customer in customers:
        if customer["customer_id"] == customer_id:
            return customer
    return None

# Not used - testing showed it hurt retrieval accuracy by pulling in irrelevant chunks based on brand/product terms
def build_enriched_query(query: str, customer: dict) -> str: # Enrich the original query with structured customer profile fields.
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


def load_vectorstore(collection_name: str):
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=collection_name,
        persist_directory=os.path.join(CHROMA_DIR, collection_name),
        embedding_function=embeddings,
    )


def retrieve_persona_context(query: str, customer_id: str, persona_vectorstore) -> tuple[str, list[str]]:
    results = persona_vectorstore.similarity_search(
        query,
        k=PERSONA_TOP_K,
        filter={"customer_id": customer_id},
    )
    chunks = [doc.page_content for doc in results]
    context = "\n".join(chunks)
    return context, chunks


def run(query: str, customer_id: str, vectorstore=None, persona_vectorstore=None) -> dict:
    start = time.perf_counter()

    if vectorstore is None:
        vectorstore = load_vectorstore("knowledge_base")
    if persona_vectorstore is None:
        persona_vectorstore = load_vectorstore("customer_personas")

    # Load structured customer profile
    customer = load_customer(customer_id)
    if not customer:
        return {"error": f"Customer {customer_id} not found."}

    persona_context, persona_chunks = retrieve_persona_context(query, customer_id, persona_vectorstore)

    enriched_query = build_enriched_query(query, customer)

    # Retrieve top-k knowledge base chunks
    results = vectorstore.similarity_search(query, k=TOP_K)
    retrieved_chunks = [doc.page_content for doc in results]

    context = shared_pipeline(query, retrieved_chunks)

    # Combine persona context and knowledge base context
    full_context = f"Customer Profile\n{persona_context}\n\nKnowledge Base\n{context}"

    # Generate personalised response
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
        "persona_chunks": persona_chunks,
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
