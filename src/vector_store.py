import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


def get_embeddings():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key != "cole-sua-chave-aqui":
        return OpenAIEmbeddings(openai_api_key=api_key)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(collection_name: str = "fintech_docs") -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def ingest_documents(
    documents: List[Document],
    collection_name: str = "fintech_docs",
) -> Chroma:
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(documents)
    print(f"  {len(documents)} chunks indexados no ChromaDB")
    return vector_store


def count_documents(collection_name: str = "fintech_docs") -> int:
    vector_store = get_vector_store(collection_name)
    return vector_store._collection.count()


if __name__ == "__main__":
    from loader import load_documents_from_dir
    from chunker import chunk_documents

    docs_dir = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
    print("Carregando documentos...")
    docs = load_documents_from_dir(docs_dir)
    print(f"Total: {len(docs)} documentos/páginas")

    print("Dividindo em chunks...")
    chunks = chunk_documents(docs)
    print(f"Total: {len(chunks)} chunks")

    print("Inserindo no ChromaDB...")
    ingest_documents(chunks)

    total = count_documents()
    print(f"\nPronto! {total} chunks indexados em data/chroma_db/")
