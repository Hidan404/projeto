from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def dividir_documentos(
    documentos: List[Document],
    tamanho_chunk: int = 1000,
    sobreposicao_chunk: int = 200,
) -> List[Document]:
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamanho_chunk,
        chunk_overlap=sobreposicao_chunk,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return divisor.split_documents(documentos)


if __name__ == "__main__":
    from loader import carregar_documento, carregar_documentos_do_diretorio
    import sys, os

    alvo = sys.argv[1] if len(sys.argv) > 1 else "../data/documents"
    if os.path.isdir(alvo):
        docs = carregar_documentos_do_diretorio(alvo)
    else:
        docs = carregar_documento(alvo)

    pedacos = dividir_documentos(docs)
    print(f"Documentos carregados: {len(docs)}")
    print(f"Chunks gerados: {len(pedacos)}")
    if pedacos:
        tamanho_medio = sum(len(p.page_content) for p in pedacos) / len(pedacos)
        print(f"Tamanho medio dos chunks: {tamanho_medio:.0f} caracteres")
        for p in pedacos[:3]:
            print(f"\n--- Chunk (fonte: {p.metadata.get('source', '?')}) ---")
            print(p.page_content[:200])
