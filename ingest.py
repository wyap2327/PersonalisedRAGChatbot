"""
Loads all knowledge base documents and customer persona files, chunks and embeds them using BAAI/bge-base-en-v1.5, 
and stores them in ChromaDB with metadata for filtering.

Run: python ingest.py
Requirements: pip install langchain langchain-community chromadb sentence-transformers
"""

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "Knowledge_base")
PERSONAS_DIR = os.path.join(BASE_DIR, "Customer & Persona", "customer_data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Load knowledge base documents with metadata: source_type, category (FAQ/Policies/Products), filename
def load_knowledge_base(): 
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

# Load customer persona files with metadata: source_type, customer_id, filename
def load_persona_files():
    documents = []
    for file in sorted(os.listdir(PERSONAS_DIR)):
        if not file.endswith(".txt"):
            continue
        filepath = os.path.join(PERSONAS_DIR, file)
        customer_id = os.path.splitext(file)[0]    # e.g. CUST-001 from CUST-001.txt
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

# Split documents into chunks using RecursiveCharacterTextSplitter
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\n  Total chunks created: {len(chunks)}")
    return chunks

# Embed chunks and store in ChromaDB
def build_vectorstore(chunks, collection_name):
    collection_dir = os.path.join(CHROMA_DIR, collection_name)
    if os.path.exists(collection_dir):
        shutil.rmtree(collection_dir)

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
    print("Loading knowledge base documents...")
    kb_docs = load_knowledge_base()

    print("Chunking knowledge base documents...")
    kb_chunks = chunk_documents(kb_docs)

    print("Embedding and storing knowledge base...")
    build_vectorstore(kb_chunks, collection_name="knowledge_base")

    print("Loading and storing customer persona files...")
    persona_docs = load_persona_files()
    persona_chunks = chunk_documents(persona_docs)
    build_vectorstore(persona_chunks, collection_name="customer_personas")

    print("Ingestion complete.")

if __name__ == "__main__":
    main()
