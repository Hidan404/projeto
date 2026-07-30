"""Testes de integracao: pipeline completo loader -> chunker -> QA.

Necessita de chave Groq configurada no .env para os testes de QA.
"""

import os
import sys

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from chunker import dividir_documentos
from loader import carregar_documento
from retriever import criar_cadeia_qa, perguntar
from vector_store import (
    obter_banco_vetorial, ingerir_documentos, contar_documentos,
    ingerir_documentos_no_startup,
)

DIR_DOCUMENTS = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
DIR_TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")


class TestPipelineIntegracao:
    """Testes que validam o pipeline completo com os dados reais."""

    def test_loader_chunker_pipeline(self):
        """Carrega um documento real, divide em chunks e verifica."""
        docs = carregar_documento(os.path.join(DIR_DOCUMENTS, "nubank_aviso_privacidade.txt"))
        assert len(docs) == 1
        assert len(docs[0].page_content) > 1000

        chunks = dividir_documentos(docs)
        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.page_content) > 0
            assert "nubank" in c.metadata.get("source", "").lower()

    @pytest.mark.slow
    def test_qa_resposta_valida(self):
        """Testa se o RAG retorna uma resposta nao vazia para pergunta valida."""
        cadeia, rec = criar_cadeia_qa()
        ingerir_documentos_no_startup()
        resposta, docs = perguntar(
            "Quais dados pessoais o Nubank coleta?",
            cadeia=cadeia,
            recuperador=rec,
        )
        assert resposta, "Resposta nao pode ser vazia"
        assert len(resposta) > 20, "Resposta muito curta"

    @pytest.mark.slow
    def test_qa_fora_de_contexto(self):
        """Pergunta sem contexto deve retornar mensagem padrao."""
        cadeia, rec = criar_cadeia_qa()
        ingerir_documentos_no_startup()
        resposta, docs = perguntar(
            "Qual a previsao do tempo para amanha?",
            cadeia=cadeia,
            recuperador=rec,
        )
        # Deve reconhecer que nao tem a informacao (aceita acentos)
        resposta_normalizada = resposta.lower().replace("ã", "a").replace("õ", "o").replace("é", "e")
        assert "nao encontrei" in resposta_normalizada or "nao encontrada" in resposta_normalizada

    @pytest.mark.slow
    def test_qa_cita_fonte(self):
        """A resposta deve mencionar o nome do documento fonte."""
        cadeia, rec = criar_cadeia_qa()
        ingerir_documentos_no_startup()
        resposta, docs = perguntar(
            "Qual a politica do Nubank para dados biometricos?",
            cadeia=cadeia,
            recuperador=rec,
        )
        assert "nubank" in resposta.lower() or "Documento" in resposta

    @pytest.mark.slow
    def test_recuperador_retorna_k_docs(self):
        """O recuperador deve retornar exatamente k documentos."""
        cadeia, rec = criar_cadeia_qa()
        ingerir_documentos_no_startup()
        docs = rec.invoke("politica de privacidade")
        assert len(docs) == 6, f"Esperado 6, obtido {len(docs)}"

    def test_contar_documentos_reais(self):
        """Verifica se ha documentos indexados no ChromaDB apos startup."""
        ingerir_documentos_no_startup()
        total = contar_documentos()
        assert total > 0, "Nenhum documento indexado no ChromaDB"
        print(f"  Total de chunks indexados: {total}")

    def test_listar_documentos(self):
        """Verifica se lista os documentos disponiveis."""
        from retriever import listar_documentos_disponiveis
        docs = listar_documentos_disponiveis()
        assert len(docs) > 0
        assert any("nubank" in d.lower() for d in docs)


class TestPipelineFormatos:
    """Testa o pipeline completo para cada formato suportado."""

    @pytest.mark.parametrize("arquivo", [
        "sample.txt", "sample.md", "sample.csv",
        "sample.json", "sample.html", "sample.pdf",
        "sample.docx", "sample.pptx", "sample.xlsx",
    ])
    def test_formato_no_pipeline(self, arquivo):
        """Cada formato deve passar pelo loader + chunker sem erro."""
        caminho = os.path.join(DIR_TEST_DATA, arquivo)
        if not os.path.exists(caminho):
            pytest.skip(f"Arquivo {arquivo} nao encontrado")
        docs = carregar_documento(caminho)
        assert len(docs) > 0
        chunks = dividir_documentos(docs)
        assert len(chunks) > 0
        for c in chunks:
            assert len(c.page_content) > 0
            assert c.metadata.get("source", "").endswith(arquivo)
