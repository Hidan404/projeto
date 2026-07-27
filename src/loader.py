import os
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


def carregar_documento(caminho_arquivo: str) -> List[Document]:
    ext = os.path.splitext(caminho_arquivo)[1].lower()
    mapa_ext = {
        ".pdf": _carregar_pdf,
        ".txt": _carregar_texto,
        ".md": _carregar_markdown,
        ".csv": _carregar_csv,
        ".json": _carregar_json,
        ".docx": _carregar_docx,
        ".pptx": _carregar_pptx,
        ".html": _carregar_html,
        ".htm": _carregar_html,
        ".xlsx": _carregar_xlsx,
        ".xls": _carregar_xlsx,
    }
    func_carregador = mapa_ext.get(ext)
    if not func_carregador:
        raise ValueError(f"Formato nao suportado: {ext}")
    return func_carregador(caminho_arquivo)


def carregar_documentos_do_diretorio(diretorio: str) -> List[Document]:
    todos_docs = []
    for raiz, _, arquivos in os.walk(diretorio):
        for nome_arquivo in arquivos:
            caminho = os.path.join(raiz, nome_arquivo)
            try:
                docs = carregar_documento(caminho)
                todos_docs.extend(docs)
                print(f"  OK  {nome_arquivo} ({len(docs)} paginas/blocos)")
            except Exception as e:
                print(f"  ERRO {nome_arquivo}: {e}")
    return todos_docs


def _carregar_pdf(caminho: str) -> List[Document]:
    carregador = PyPDFLoader(caminho)
    return carregador.load()


def _carregar_texto(caminho: str) -> List[Document]:
    carregador = TextLoader(caminho, encoding="utf-8")
    return carregador.load()


def _carregar_markdown(caminho: str) -> List[Document]:
    carregador = UnstructuredMarkdownLoader(caminho)
    return carregador.load()


def _carregar_csv(caminho: str) -> List[Document]:
    carregador = CSVLoader(caminho, encoding="utf-8")
    return carregador.load()


def _carregar_json(caminho: str) -> List[Document]:
    carregador = JSONLoader(caminho, jq_schema=".", text_content=False)
    return carregador.load()


def _carregar_docx(caminho: str) -> List[Document]:
    carregador = Docx2txtLoader(caminho)
    return carregador.load()


def _carregar_pptx(caminho: str) -> List[Document]:
    carregador = UnstructuredPowerPointLoader(caminho)
    return carregador.load()


def _carregar_html(caminho: str) -> List[Document]:
    carregador = UnstructuredHTMLLoader(caminho)
    return carregador.load()


def _carregar_xlsx(caminho: str) -> List[Document]:
    import pandas as pd
    docs = []
    planilha = pd.ExcelFile(caminho)
    for nome_aba in planilha.sheet_names:
        df = pd.read_excel(caminho, sheet_name=nome_aba)
        conteudo = df.to_string(index=False)
        doc = Document(
            page_content=conteudo,
            metadata={"source": caminho, "sheet": nome_aba},
        )
        docs.append(doc)
    return docs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python loader.py <arquivo-ou-diretorio>")
        sys.exit(1)
    alvo = sys.argv[1]
    if os.path.isdir(alvo):
        docs = carregar_documentos_do_diretorio(alvo)
    else:
        docs = carregar_documento(alvo)
    print(f"\nTotal de documentos carregados: {len(docs)}")
    for d in docs[:3]:
        print(f"  -> {len(d.page_content)} chars | fonte: {d.metadata.get('source', '?')}")
