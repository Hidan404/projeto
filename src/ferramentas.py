"""Ferramentas que o agente pode usar: documentos (RAG) e banco de dados (SQL)."""

from typing import List, Tuple

from langchain_core.tools import Tool

from banco import inicializar_banco, perguntar_banco, resumo_banco
from retriever import criar_cadeia_qa, perguntar as perguntar_rag

inicializar_banco()
_cadeia, _recuperador = criar_cadeia_qa()
_DESCRICAO_BANCO = resumo_banco()


def consultar_documentos(pergunta: str) -> Tuple[str, str, List[str]]:
    """Retorna (resposta_texto, nome_ferramenta, lista_fontes)."""
    resultado, docs = perguntar_rag(pergunta, cadeia=_cadeia, recuperador=_recuperador)
    fontes = set()
    for doc in docs if docs else []:
        nome = doc.metadata.get("source", "").split("/")[-1]
        if nome:
            fontes.add(nome)
    return resultado, "Documentos", sorted(fontes)


def consultar_banco(pergunta: str) -> Tuple[str, str, List[str]]:
    """Retorna (resposta_texto, nome_ferramenta, lista_fontes)."""
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
    description=(
        "Usar para perguntas sobre dados estruturados: "
        "clientes, empresas, CNPJ, faturamento, planos, segmentos, "
        "produtos financeiros, taxas de juros, creditos, cartoes, "
        "transacoes, movimentacoes, entradas, saidas, valores totais. "
        "O banco tem as tabelas: clientes, produtos_financeiros, transacoes.\n"
        f"Schema:\n{_DESCRICAO_BANCO}"
    ),
)

ferramentas_disponiveis = [ferramenta_documentos, ferramenta_banco]
"""Lista de ferramentas disponiveis para o agente."""


def classificar_pergunta(pergunta: str) -> List[Tool]:
    """Classifica a pergunta para determinar quais ferramentas usar."""
    p = pergunta.lower()

    palavras_documentos = [
        "politica", "privacidade", "contrato", "termo", "biometrico",
        "nubank", "stripe", "picpay", "mercadopago", "documento",
        "lgpd", "dado pessoal", "seguranca", "clausula", "procedimento",
    ]
    palavras_banco = [
        "cliente", "empresa", "cnpj", "faturamento", "plano", "segmento",
        "produto", "servico", "credito", "cartao", "taxa", "juro",
        "transac", "movimentac", "entrada", "saida", "receita", "despesa",
        "total", "maior", "menor", "medio", "top", "ranking", "giro",
        "conta digital", "maquininha", "cambio", "antecipac",
    ]

    usar_docs = any(palavra in p for palavra in palavras_documentos)
    usar_banco = any(palavra in p for palavra in palavras_banco)

    ferramentas_selecionadas = []
    if usar_docs:
        ferramentas_selecionadas.append(ferramenta_documentos)
    if usar_banco:
        ferramentas_selecionadas.append(ferramenta_banco)
    if not ferramentas_selecionadas:
        ferramentas_selecionadas.append(ferramenta_documentos)

    return ferramentas_selecionadas


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
