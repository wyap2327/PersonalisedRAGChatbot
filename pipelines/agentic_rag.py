import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import create_react_agent
from pipelines.llm import MODEL, generate_response
from pipelines.shared_pipeline import shared_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "Knowledge_base")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K = 5

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
    vs = _get_vectorstore()
    results = vs.similarity_search(query, k=TOP_K)
    chunks = [doc.page_content for doc in results]
    return "\n\n---\n\n".join(chunks) if chunks else "No relevant information found."


@tool
def GetFullDocument(filename: str) -> str:
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        if filename in files:
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
    return f"Document '{filename}' not found."


def run(query: str, vectorstore=None) -> dict:
    start = time.perf_counter()

    if vectorstore is not None:
        global _vectorstore
        _vectorstore = vectorstore

    tools = [SearchKnowledgeBase, GetFullDocument]
    llm = ChatOllama(model=MODEL, temperature=0)
    agent = create_react_agent(llm, tools)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 10},
    )

    tool_outputs = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]

    if tool_outputs:
        # SearchKnowledgeBase joins chunks with "\n\n---\n\n"; GetFullDocument returns plain text.
        all_chunks = []
        for output in tool_outputs:
            if "\n\n---\n\n" in output:
                all_chunks.extend(c.strip() for c in output.split("\n\n---\n\n") if c.strip())
            elif output.strip() and output.strip() != "No relevant information found.":
                all_chunks.append(output.strip())

        context = shared_pipeline(query, all_chunks) if all_chunks else "No relevant information found."
        response = generate_response(query=query, context=context)
    else:
        # Agent answered without calling any tools — use its response directly
        context = "No tool calls made."
        response = result["messages"][-1].content

    latency = round(time.perf_counter() - start, 3)

    return {
        "pipeline": "agentic_rag",
        "query": query,
        "response": response,
        "context": context,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    query = "I bought a jacket 3 weeks ago and it has a broken zip. What are my options?"
    print(f"Query: {query}\n")
    result = run(query)
    print(f"Response:\n{result['response']}")
    print(f"\nLatency: {result['latency_seconds']}s")
