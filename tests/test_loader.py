"""Testes unitarios para o modulo loader (src/loader.py)."""

import os
import sys
import tempfile

import pytest

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from loader import carregar_documento, carregar_documentos_do_diretorio

DIR_TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")


class TestLoader:
    def test_carregar_txt(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.txt"))
        assert len(docs) == 1
        assert "Teste TXT" in docs[0].page_content
        assert docs[0].metadata.get("source", "").endswith("sample.txt")

    def test_carregar_md(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.md"))
        assert len(docs) == 1
        assert "Teste Formatos" in docs[0].page_content

    def test_carregar_csv(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.csv"))
        assert len(docs) == 3  # 3 linhas de dados
        assert "Alice" in docs[0].page_content
        assert docs[0].metadata.get("linha") == 1

    def test_carregar_json(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.json"))
        assert len(docs) == 1
        assert "TechMock" in docs[0].page_content
        assert "departamentos" in docs[0].page_content

    def test_carregar_html(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.html"))
        assert len(docs) == 1
        assert "Teste HTML" in docs[0].page_content

    def test_carregar_pdf(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.pdf"))
        assert len(docs) >= 1
        assert "Teste PDF" in docs[0].page_content or "Documento" in docs[0].page_content

    def test_carregar_docx(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.docx"))
        assert len(docs) == 1
        assert "Teste DOCX" in docs[0].page_content

    def test_carregar_pptx(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.pptx"))
        assert len(docs) == 1
        assert "Teste PPTX" in docs[0].page_content or "Teste" in docs[0].page_content

    def test_carregar_xlsx(self):
        docs = carregar_documento(os.path.join(DIR_TEST_DATA, "sample.xlsx"))
        assert len(docs) >= 1
        # Deve conter dados da planilha
        conteudo = docs[0].page_content
        assert "Nome" in conteudo or "Item A" in conteudo

    def test_formato_invalido(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"teste")
            caminho = f.name
        try:
            with pytest.raises(ValueError, match="nao suportado"):
                carregar_documento(caminho)
        finally:
            os.unlink(caminho)

    def test_diretorio_com_erro(self):
        """Diretorio com um arquivo valido e um invalido deve continuar."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Arquivo valido
            with open(os.path.join(tmpdir, "ok.txt"), "w") as f:
                f.write("conteudo valido")
            # Arquivo invalido
            with open(os.path.join(tmpdir, "bad.xyz"), "w") as f:
                f.write("conteudo invalido")
            docs = carregar_documentos_do_diretorio(tmpdir)
            assert len(docs) == 1

    def test_diretorio_vazio(self):
        docs = carregar_documentos_do_diretorio("/tmp/nao_existe_xyz")
        assert docs == []
