from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    from loader import load_document, load_documents_from_dir
    import sys, os

    target = sys.argv[1] if len(sys.argv) > 1 else "../data/documents"
    if os.path.isdir(target):
        docs = load_documents_from_dir(target)
    else:
        docs = load_document(target)

    chunks = chunk_documents(docs)
    print(f"Documentos carregados: {len(docs)}")
    print(f"Chunks gerados: {len(chunks)}")
    print(f"Tamanho médio dos chunks: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    for c in chunks[:3]:
        print(f"\n--- Chunk (fonte: {c.metadata.get('source', '?')}) ---")
        print(c.page_content[:200])
