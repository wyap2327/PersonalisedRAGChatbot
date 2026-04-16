"""
llm.py

Wrapper for calling the local Ollama LLM (llama3.1:8b).
All 5 RAG pipelines use this to generate the final response.
"""

import ollama

MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a helpful customer support assistant for ShopNest, \
a UK-based sportswear and footwear e-commerce store.

Answer the customer's question using ONLY the information provided in the context below.
If the answer is not in the context, say: "I'm sorry, I don't have that information. \
Please contact our support team at support@shopnest.co.uk or call 0800 123 4567."

Be concise, friendly, and professional. Do not make up information."""


def generate_response(query: str, context: str, customer_name: str = None) -> str:
    """
    Generate a customer support response using the local LLM.

    Args:
        query         : the customer's question
        context       : retrieved and processed context from the shared pipeline
        customer_name : optional, used by Contextual RAG to personalise the greeting

    Returns:
        The LLM's response as a string.
    """
    greeting = f"You are speaking with {customer_name}. " if customer_name else ""

    user_message = f"""{greeting}

Context:
{context}

Customer question: {query}"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    return response["message"]["content"]
