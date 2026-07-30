import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DIR_DOCS = os.path.join(os.path.dirname(__file__), "..", "data", "documents")


def obter_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def obter_banco_vetorial(nome_colecao: str = "fintech_docs") -> Chroma:
    embeddings = obter_embeddings()
    return Chroma(
        collection_name=nome_colecao,
        embedding_function=embeddings,
    )


def ingerir_documentos(
    documentos: List[Document],
    nome_colecao: str = "fintech_docs",
) -> Chroma:
    banco_vetorial = obter_banco_vetorial(nome_colecao)
    banco_vetorial.add_documents(documentos)
    return banco_vetorial


def contar_documentos(nome_colecao: str = "fintech_docs") -> int:
    banco_vetorial = obter_banco_vetorial(nome_colecao)
    return len(banco_vetorial.get()["ids"])


def ingerir_documentos_no_startup(nome_colecao: str = "fintech_docs") -> Chroma:
    """Carrega, chunk e indexa todos os documentos no ChromaDB em memoria."""
    from loader import carregar_documentos_do_diretorio
    from chunker import dividir_documentos

    docs = carregar_documentos_do_diretorio(DIR_DOCS)
    pedacos = dividir_documentos(docs)
    banco_vetorial = obter_banco_vetorial(nome_colecao)
    banco_vetorial.add_documents(pedacos)
    return banco_vetorial


if __name__ == "__main__":
    total = contar_documentos()
    print(f"Chunks indexados: {total}")
