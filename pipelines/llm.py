import ollama

MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a helpful customer support assistant for ShopNest, \
a UK-based sportswear and footwear e-commerce store.

Answer the customer's question using ONLY the information provided in the context below.
If the answer is not in the context, say: "I'm sorry, I don't have that information. \
Please contact our support team at support@shopnest.co.uk or call 0800 123 4567."

Be concise, friendly, and professional. Do not make up information."""


def generate_response(query: str, context: str, customer_name: str = None) -> str:
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
        options={"temperature": 0},
    )
    return response["message"]["content"]
