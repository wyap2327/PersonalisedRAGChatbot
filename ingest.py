"""
ingest.py

Loads all ShopNest knowledge base documents and customer persona files,
chunks them, embeds them using BAAI/bge-base-en-v1.5, and stores them
in ChromaDB with metadata for filtering.

Two ChromaDB collections are created:
  - knowledge_base   : FAQ, policies, product catalogue
  - customer_personas: one persona file per customer (filtered by customer_id)

Run: python ingest.py
Requirements: pip install langchain langchain-community chromadb sentence-transformers
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "Knowledge_base")
PERSONAS_DIR = os.path.join(BASE_DIR, "Customer & Persona", "customer_data", "personas")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_knowledge_base():
    """
    Walk the Knowledge_base directory and load all .txt files.
    Attaches metadata: source_type, category (FAQ/Policies/Products), filename.
    """
    documents = []
    for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue
            filepath = os.path.join(root, file)
            category = os.path.basename(root)   # FAQ, Policies, or Products
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata.update({
                    "source_type": "knowledge_base",
                    "category": category,
                    "filename": file,
                })
            documents.extend(docs)
            print(f"  Loaded [{category}] {file}")
    return documents


def load_persona_files():
    """
    Load each customer persona .txt file from the personas directory.
    Attaches metadata: source_type, customer_id, filename.
    """
    documents = []
    for file in sorted(os.listdir(PERSONAS_DIR)):
        if not file.endswith(".txt"):
            continue
        filepath = os.path.join(PERSONAS_DIR, file)
        customer_id = file.split("_")[0]    # e.g. CUST-001 from CUST-001_griffiths.txt
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source_type": "persona",
                "customer_id": customer_id,
                "filename": file,
            })
        documents.extend(docs)
        print(f"  Loaded [Persona] {file} ({customer_id})")
    return documents


def chunk_documents(documents):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  Total chunks created: {len(chunks)}")
    return chunks


def build_vectorstore(chunks, collection_name):
    """Embed chunks and store in ChromaDB."""
    print(f"\n  Embedding and storing in collection: '{collection_name}'")
    print(f"  Using model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=os.path.join(CHROMA_DIR, collection_name),
    )

    print(f"  Stored {vectorstore._collection.count()} chunks in '{collection_name}'")
    return vectorstore


def main():
    print("=" * 60)
    print("ShopNest ChromaDB Ingestion")
    print("=" * 60)

    # ── Knowledge base ────────────────────────────────────────────
    print("\n[1/4] Loading knowledge base documents...")
    kb_docs = load_knowledge_base()
    print(f"      {len(kb_docs)} documents loaded")

    print("\n[2/4] Chunking knowledge base documents...")
    kb_chunks = chunk_documents(kb_docs)

    print("\n[3/4] Embedding and storing knowledge base...")
    build_vectorstore(kb_chunks, collection_name="knowledge_base")

    # ── Customer personas ─────────────────────────────────────────
    print("\n[4/4] Loading and storing customer persona files...")
    persona_docs = load_persona_files()
    persona_chunks = chunk_documents(persona_docs)
    build_vectorstore(persona_chunks, collection_name="customer_personas")

    print("\n" + "=" * 60)
    print("Ingestion complete.")
    print(f"ChromaDB stored at: {CHROMA_DIR}")
    print("Collections created:")
    print("  - knowledge_base    (FAQ, Policies, Products)")
    print("  - customer_personas (10 customer profile files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
