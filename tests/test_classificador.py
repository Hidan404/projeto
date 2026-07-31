"""Testes de regressao do classificador de perguntas (ferramentas.classificar_pergunta)."""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from ferramentas import classificar_pergunta


class TestClassificador:
    def _nomes(self, pergunta):
        return [t.name for t in classificar_pergunta(pergunta)]

    # --- Perguntas de banco de dados ---
    def test_taxa_juros_credito(self):
        assert self._nomes("Qual a taxa de juros do credito?") == ["consultar_banco_dados"]

    def test_produtos_financeiros(self):
        assert self._nomes("Quais produtos financeiros temos?") == ["consultar_banco_dados"]

    def test_valor_minimo_cartao(self):
        assert self._nomes("Qual o valor minimo do cartao?") == ["consultar_banco_dados"]

    def test_cliente_maior_faturamento(self):
        assert self._nomes("Qual cliente tem maior faturamento?") == ["consultar_banco_dados"]

    def test_servicos_financeiros(self):
        assert self._nomes("Quais servicos financeiros?") == ["consultar_banco_dados"]

    def test_listar_transacoes(self):
        assert self._nomes("Listar todas as transacoes") == ["consultar_banco_dados"]

    def test_quantos_clientes(self):
        assert self._nomes("Quantos clientes cadastrados?") == ["consultar_banco_dados"]

    def test_credito_acentuado(self):
        assert self._nomes("Qual a taxa de juros do crédito?") == ["consultar_banco_dados"]

    # --- Perguntas de documentos ---
    def test_politica_privacidade(self):
        assert self._nomes("Qual a politica de privacidade do Nubank?") == ["consultar_documentos"]

    def test_limite_credito_mercado_pago(self):
        assert self._nomes(
            "Como funciona o limite de credito com reserva no Mercado Pago?"
        ) == ["consultar_documentos"]

    def test_juros_rotativo(self):
        assert self._nomes("Quais os juros do rotativo?") == ["consultar_documentos"]

    def test_limites_transacao_dia(self):
        assert self._nomes("Quais sao os limites de transacao por dia?") == ["consultar_documentos"]

    def test_termos_servico(self):
        assert self._nomes("Termos de servico do Stripe") == ["consultar_documentos"]

    def test_fatura_nao_colide_com_faturamento(self):
        # "fatura" (docs) nao deve casar dentro de "faturamento"
        assert self._nomes("Qual o faturamento medio dos clientes?") == ["consultar_banco_dados"]
