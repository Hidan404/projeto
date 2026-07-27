import os
import tempfile
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        ".pdf": _load_pdf,
        ".txt": _load_text,
        ".md": _load_markdown,
        ".csv": _load_csv,
        ".json": _load_json,
        ".docx": _load_docx,
        ".pptx": _load_pptx,
        ".html": _load_html,
        ".htm": _load_html,
        ".xlsx": _load_xlsx,
        ".xls": _load_xlsx,
    }
    loader_fn = ext_map.get(ext)
    if not loader_fn:
        raise ValueError(f"Formato não suportado: {ext}")
    return loader_fn(file_path)


def load_documents_from_dir(directory: str) -> List[Document]:
    all_docs = []
    for root, _, files in os.walk(directory):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                docs = load_document(path)
                all_docs.extend(docs)
                print(f"  OK  {fname} ({len(docs)} páginas/chunks)")
            except Exception as e:
                print(f"  ERRO {fname}: {e}")
    return all_docs


def _load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


def _load_text(path: str) -> List[Document]:
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()


def _load_markdown(path: str) -> List[Document]:
    loader = UnstructuredMarkdownLoader(path)
    return loader.load()


def _load_csv(path: str) -> List[Document]:
    loader = CSVLoader(path, encoding="utf-8")
    return loader.load()


def _load_json(path: str) -> List[Document]:
    loader = JSONLoader(path, jq_schema=".", text_content=False)
    return loader.load()


def _load_docx(path: str) -> List[Document]:
    loader = Docx2txtLoader(path)
    return loader.load()


def _load_pptx(path: str) -> List[Document]:
    loader = UnstructuredPowerPointLoader(path)
    return loader.load()


def _load_html(path: str) -> List[Document]:
    loader = UnstructuredHTMLLoader(path)
    return loader.load()


def _load_xlsx(path: str) -> List[Document]:
    import pandas as pd
    docs = []
    xl = pd.ExcelFile(path)
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        content = df.to_string(index=False)
        doc = Document(
            page_content=content,
            metadata={"source": path, "sheet": sheet_name},
        )
        docs.append(doc)
    return docs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python loader.py <arquivo-ou-diretório>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        docs = load_documents_from_dir(target)
    else:
        docs = load_document(target)
    print(f"\nTotal de documentos carregados: {len(docs)}")
    for d in docs[:3]:
        print(f"  → {len(d.page_content)} chars | fonte: {d.metadata.get('source', '?')}")
