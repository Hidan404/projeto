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
    """Classifica a pergunta para determinar quais ferramentas usar.

    - Se a pergunta mencionar termos de documentos (politica, privacidade,
      contrato, etc.), usa APENAS documentos.
    - Se mencionar termos de banco de dados (cliente, faturamento, transacao,
      etc.) sem termos de documento, usa APENAS banco.
    - Soh combina os dois quando ha indicadores fortes de ambos os dominios
      (ex: "qual a politica de dados dos clientes ativos?").
    """
    pergunta_lower = pergunta.lower()

    # Palavras que indicam pergunta SOBRE documentos (politicas, contratos)
    palavras_docs_forte = [
        "politica", "privacidade", "contrato", "termo", "biometrico",
        "nubank", "stripe", "picpay", "mercadopago", "documento",
        "lgpd", "dado pessoal", "clausula",
    ]
    # Palavras que indicam pergunta SOBRE dados estruturados
    palavras_banco_forte = [
        "faturamento", "segmento", "cnpj", "transac", "movimentac",
        "entrada", "saida", "receita", "despesa", "total",
        "maior", "menor", "medio", "top", "ranking", "giro",
        "taxa", "juro", "conta digital", "maquininha", "cambio",
        "antecipac", "plano",
    ]
    # Palavras ambivalentes - so ativam banco se nao houver termos de documento
    palavras_banco_fraco = [
        "cliente", "empresa", "produto", "servico", "credito", "cartao",
    ]

    tem_docs = any(palavra in pergunta_lower for palavra in palavras_docs_forte)
    tem_banco_forte = any(palavra in pergunta_lower for palavra in palavras_banco_forte)
    tem_banco_fraco = any(palavra in pergunta_lower for palavra in palavras_banco_fraco)

    ferramentas_selecionadas = []

    if tem_docs:
        # Prioritariamente documentos
        ferramentas_selecionadas.append(ferramenta_documentos)
        # Soh adiciona banco se houver termo FORTE de banco (nao apenas "cliente")
        if tem_banco_forte:
            ferramentas_selecionadas.append(ferramenta_banco)
    elif tem_banco_forte or tem_banco_fraco:
        ferramentas_selecionadas.append(ferramenta_banco)
    else:
        # Fallback: documentos
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
