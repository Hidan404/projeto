import os
from typing import List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

DIR_PERSISTENCIA = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")


def obter_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def obter_banco_vetorial(nome_colecao: str = "fintech_docs") -> Chroma:
    embeddings = obter_embeddings()
    return Chroma(
        collection_name=nome_colecao,
        embedding_function=embeddings,
        persist_directory=DIR_PERSISTENCIA,
    )


def ingerir_documentos(
    documentos: List[Document],
    nome_colecao: str = "fintech_docs",
) -> Chroma:
    banco_vetorial = obter_banco_vetorial(nome_colecao)
    banco_vetorial.add_documents(documentos)
    print(f"  {len(documentos)} chunks indexados no ChromaDB")
    return banco_vetorial


def contar_documentos(nome_colecao: str = "fintech_docs") -> int:
    banco_vetorial = obter_banco_vetorial(nome_colecao)
    return banco_vetorial._collection.count()


if __name__ == "__main__":
    from loader import carregar_documentos_do_diretorio
    from chunker import dividir_documentos

    dir_documentos = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
    print("Carregando documentos...")
    docs = carregar_documentos_do_diretorio(dir_documentos)
    print(f"Total: {len(docs)} documentos/paginas")

    print("Dividindo em chunks...")
    pedacos = dividir_documentos(docs)
    print(f"Total: {len(pedacos)} chunks")

    print("Inserindo no ChromaDB...")
    ingerir_documentos(pedacos)

    total = contar_documentos()
    print(f"\nPronto! {total} chunks indexados em data/chroma_db/")
