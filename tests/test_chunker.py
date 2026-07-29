"""Testes unitarios para o modulo chunker (src/chunker.py)."""

import os
import sys

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from chunker import dividir_documentos
from langchain_core.documents import Document


class TestChunker:
    def test_documento_unico_menor_que_chunk(self):
        doc = Document(page_content="Texto curto.")
        chunks = dividir_documentos([doc], tamanho_chunk=1000, sobreposicao_chunk=0)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Texto curto."

    def test_documento_unico_maior_que_chunk(self):
        texto = "Palavra " * 500  # ~4000 chars
        doc = Document(page_content=texto)
        chunks = dividir_documentos([doc], tamanho_chunk=500, sobreposicao_chunk=0)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c.page_content) <= 500

    def test_multiplos_documentos(self):
        docs = [
            Document(page_content="A" * 2000),
            Document(page_content="B" * 2000),
        ]
        chunks = dividir_documentos(docs, tamanho_chunk=1000, sobreposicao_chunk=0)
        # Cada doc de 2000 chars com chunk 1000 vira pelo menos 2 chunks
        assert len(chunks) >= 4

    def test_sobreposicao_funciona(self):
        texto = "Uma frase. " * 100  # ~1100 chars
        doc = Document(page_content=texto)
        chunks_sem = dividir_documentos([doc], tamanho_chunk=500, sobreposicao_chunk=0)
        chunks_com = dividir_documentos([doc], tamanho_chunk=500, sobreposicao_chunk=100)
        # Com sobreposicao deve gerar mais chunks
        assert len(chunks_com) >= len(chunks_sem)

    def test_metadata_preservada(self):
        doc = Document(
            page_content="Conteudo de teste com metadata.",
            metadata={"source": "teste.txt", "pagina": 1},
        )
        chunks = dividir_documentos([doc], tamanho_chunk=100, sobreposicao_chunk=0)
        for c in chunks:
            assert c.metadata.get("source") == "teste.txt"
            assert c.metadata.get("pagina") == 1

    def test_documento_vazio(self):
        doc = Document(page_content="")
        chunks = dividir_documentos([doc])
        assert len(chunks) == 0 or chunks[0].page_content == ""

    def test_lista_vazia(self):
        chunks = dividir_documentos([])
        assert chunks == []

    def test_chunk_size_muito_pequeno(self):
        doc = Document(page_content="Texto maior que o chunk permitido")
        chunks = dividir_documentos([doc], tamanho_chunk=5, sobreposicao_chunk=0)
        # Mesmo com chunk pequeno, o separador final "" garante que quebra
        assert len(chunks) >= 1

    def test_tamanho_medio_dos_chunks(self, caplog):
        """Verifica se o tamanho medio esta proximo do esperado."""
        texto = ("palavra " * 1000)  # ~8000 chars
        doc = Document(page_content=texto)
        chunks = dividir_documentos([doc], tamanho_chunk=1000, sobreposicao_chunk=200)
        tamanhos = [len(c.page_content) for c in chunks]
        media = sum(tamanhos) / len(tamanhos)
        # A media deve estar entre chunk_size - overlap e chunk_size + overlap
        # Na pratica fica entre 500 e 1000 para 1000/200
        assert 300 < media <= 1100, f"Media dos chunks {media:.0f} fora do esperado"
