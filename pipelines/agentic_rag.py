"""
agentic_rag.py

Agentic RAG pipeline.
A LangChain ReAct agent that autonomously decides how to retrieve information
using two tools:
  - SearchKnowledgeBase : semantic search over ChromaDB
  - GetFullDocument     : retrieves the full source document by filename
                          when chunks alone are insufficient

The agent reflects on retrieved results and can call tools multiple times
before generating a final answer — handling complex, multi-step queries
that simpler pipelines cannot resolve in a single retrieval step.
"""

import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from pipelines.shared_pipeline import shared_pipeline
from pipelines.llm import generate_response, MODEL

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "Knowledge_base")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5

# Global vectorstore reference used inside tools
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name="knowledge_base",
            persist_directory=os.path.join(CHROMA_DIR, "knowledge_base"),
            embedding_function=embeddings,
        )
    return _vectorstore


@tool
def SearchKnowledgeBase(query: str) -> str:
    """
    Search the ShopNest knowledge base for information relevant to the query.
    Use this for questions about policies, products, delivery, returns, or loyalty points.
    Input: a search query string.
    Output: the most relevant text passages found.
    """
    vs = _get_vectorstore()
    results = vs.similarity_search(query, k=TOP_K)
    chunks = [doc.page_content for doc in results]
    return "\n\n---\n\n".join(chunks) if chunks else "No relevant information found."


@tool
def GetFullDocument(filename: str) -> str:
    """
    Retrieve the full text of a ShopNest knowledge base document by filename.
    Use this when chunks from SearchKnowledgeBase are insufficient or incomplete.
    Available filenames: faq.txt, delivery_shipping.txt, loyalty_points.txt,
    returns_refunds.txt, product_catalogue.txt
    Input: the exact filename (e.g. 'returns_refunds.txt').
    Output: the full document text.
    """
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        if filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    return f"Document '{filename}' not found."


REACT_PROMPT = PromptTemplate.from_template("""You are a customer support assistant for ShopNest, \
a UK sportswear and footwear e-commerce store.
Answer the customer's question using the tools available to you.
Think step by step. Use tools to find relevant information before answering.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: think about what information you need
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat up to 3 times)
Thought: I now have enough information to answer the question
Final Answer: the final answer to the customer's question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


def run(query: str, vectorstore=None) -> dict:
    """
    Run the Agentic RAG pipeline.

    Args:
        query       : the customer's question
        vectorstore : optional preloaded ChromaDB instance

    Returns:
        dict with keys: response, context, latency_seconds, agent_steps
    """
    start = time.perf_counter()

    if vectorstore is not None:
        global _vectorstore
        _vectorstore = vectorstore

    tools = [SearchKnowledgeBase, GetFullDocument]

    llm = Ollama(model=MODEL, temperature=0)

    agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=4,
        handle_parsing_errors=True,
    )

    result = agent_executor.invoke({"input": query})
    agent_response = result.get("output", "")

    # Run retrieved content through shared pipeline for consistency
    retrieved = SearchKnowledgeBase.invoke(query)
    chunks = retrieved.split("\n\n---\n\n")
    context = shared_pipeline(query, chunks)

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "agentic_rag",
        "query": query,
        "response": agent_response,
        "context": context,
        "retrieved_chunks": chunks,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "I bought a jacket 3 weeks ago and it has a broken zip. What are my options?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
