"""Ferramentas que o agente pode usar: documentos (RAG) e banco de dados (SQL)."""

import re
import unicodedata
from typing import List, Tuple

from langchain_core.tools import Tool

from banco import inicializar_banco, perguntar_banco, resumo_banco
from retriever import criar_cadeia_qa, perguntar as perguntar_rag
from vector_store import ingerir_documentos_no_startup

# Inicializacao lazy — sem side effects no import
_inicializado = False
_cadeia = None
_recuperador = None


def inicializar_ferramentas():
    global _inicializado, _cadeia, _recuperador
    if _inicializado:
        return
    inicializar_banco()
    _cadeia, _recuperador = criar_cadeia_qa()
    ingerir_documentos_no_startup()
    descricao_banco = resumo_banco()
    ferramenta_banco.description = (
        "Usar para perguntas sobre dados estruturados: "
        "clientes, empresas, CNPJ, faturamento, planos, segmentos, "
        "produtos financeiros, taxas de juros, creditos, cartoes, "
        "transacoes, movimentacoes, entradas, saidas, valores totais. "
        "O banco tem as tabelas: clientes, produtos_financeiros, transacoes.\n"
        f"Schema:\n{descricao_banco}"
    )
    _inicializado = True


def consultar_documentos(pergunta: str) -> Tuple[str, str, List[str]]:
    """Retorna (resposta_texto, nome_ferramenta, lista_fontes)."""
    inicializar_ferramentas()
    resultado, docs = perguntar_rag(pergunta, cadeia=_cadeia, recuperador=_recuperador)
    fontes = set()
    for doc in docs if docs else []:
        nome = doc.metadata.get("source", "").split("/")[-1]
        if nome:
            fontes.add(nome)
    return resultado, "Documentos", sorted(fontes)


def consultar_banco(pergunta: str) -> Tuple[str, str, List[str]]:
    """Retorna (resposta_texto, nome_ferramenta, lista_fontes)."""
    inicializar_ferramentas()
    texto = perguntar_banco(pergunta)
    return texto, "Banco de Dados", ["Sistema Interno (SQL)"]


def _tool_docs(pergunta: str) -> str:
    return consultar_documentos(pergunta)[0]


def _tool_banco(pergunta: str) -> str:
    return consultar_banco(pergunta)[0]


ferramenta_documentos = Tool(
    name="consultar_documentos",
    func=_tool_docs,
    description=(
        "Usar para perguntas sobre documentos internos da fintech: "
        "politicas de privacidade, contratos, termos de servico, "
        "dados biometricos, seguranca, procedimentos. "
        "Recebe uma pergunta em portugues e retorna resposta baseada nos documentos."
    ),
)

ferramenta_banco = Tool(
    name="consultar_banco_dados",
    func=_tool_banco,
    description="Inicializacao pendente... (chame inicializar_ferramentas() primeiro)",
)

ferramentas_disponiveis = [ferramenta_documentos, ferramenta_banco]
"""Lista de ferramentas disponiveis para o agente."""


def _normalizar(texto: str) -> str:
    """Remove acentos e converte para minusculas."""
    texto = texto.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def _termo_na_pergunta(termo: str, p: str) -> bool:
    """Casa termos como palavra inteira (plural opcional); frases via substring."""
    if " " in termo:
        return termo in p
    return bool(re.search(rf"\b{re.escape(termo)}s?\b", p))


def classificar_pergunta(pergunta: str) -> List[Tool]:
    """Classifica a pergunta para uma UNICA ferramenta.

    - Termo de documento (politica, privacidade, contrato) → documentos
    - Termo de banco (cliente, produto, transacao) → banco de dados
    - Documentos tem prioridade sobre banco em caso de ambiguidade
    - Se nenhum termo for detectado, usa documentos (fallback)
    """
    p = _normalizar(pergunta)

    palavras_documentos = [
        "politica", "privacidade", "contrato", "termo", "biometrico",
        "nubank", "stripe", "picpay", "mercadopago", "documento",
        "lgpd", "dado pessoal", "clausula", "procedimento",
        "limite", "reserva", "parcelamento", "fatura",
        "cancelar", "contratar", "adesao", "juros do rotativo",
    ]
    palavras_banco = [
        "cliente", "empresa", "cnpj", "faturamento", "segmento",
        "transacao", "transacoes", "movimentacao", "movimentacoes",
        "entrada", "saida", "receita", "despesa",
        "total", "totais", "maior", "menor", "media", "medio",
        "top", "ranking", "giro",
        "conta digital", "maquininha", "cambio",
        "antecipacao", "antecipacoes",
        "cadastrados", "listar", "quantos",
        "produto", "credito", "taxa", "juros", "servico",
        "valor", "valores", "cartao", "cartoes", "plano", "emprestimo",
    ]

    tem_docs = any(_termo_na_pergunta(t, p) for t in palavras_documentos)
    tem_banco = any(_termo_na_pergunta(t, p) for t in palavras_banco)

    if tem_docs:
        return [ferramenta_documentos]
    if tem_banco:
        return [ferramenta_banco]
    return [ferramenta_documentos]


def executar_ferramentas(pergunta: str) -> List[Tuple[str, str, List[str]]]:
    """Executa as ferramentas relevantes para a pergunta.

    Retorna lista de tuplas (resposta, nome_ferramenta, fontes).
    """
    ferramentas = classificar_pergunta(pergunta)
    resultados = []
    for ferr in ferramentas:
        if ferr.name == "consultar_documentos":
            resp, nome, fontes = consultar_documentos(pergunta)
        elif ferr.name == "consultar_banco_dados":
            resp, nome, fontes = consultar_banco(pergunta)
        else:
            resp, nome, fontes = ferr.func(pergunta), ferr.name, []
        resultados.append((resp, nome, fontes))
    return resultados
