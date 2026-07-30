import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_openai import ChatOpenAI

from vector_store import obter_banco_vetorial

URL_GROQ = "https://api.groq.com/openai/v1"
MODELO_PADRAO = "llama-3.3-70b-versatile"

PROMPT_SISTEMA = """
Voce e um assistente virtual especializado em documentos internos de uma fintech.
Use APENAS o contexto fornecido abaixo para responder a pergunta do colaborador.

REGRAS:
1. Se a resposta estiver no contexto, responda com precisao usando as informacoes fornecidas.
2. Se a resposta NAO estiver claramente no contexto, diga exatamente:
   "Nao encontrei essa informacao nos documentos disponiveis."
   Nao invente nem complete informacoes que nao estao no contexto.
3. Sempre mencione ao final o nome do documento de onde a informacao foi extraida.
4. Se o contexto mencionar multiplos documentos, indique qual deles contem a resposta.
5. Responda em portugues claro e objetivo.

Contexto:
{context}

Pergunta: {input}

Resposta:
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", PROMPT_SISTEMA),
    ("human", "{input}"),
])


def _formatar_documentos(documentos: List[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in documentos)


def criar_cadeia_qa(modelo: str = "", temperatura: float = 0.0):
    load_dotenv()
    chave_api = os.getenv("OPENAI_API_KEY", "")
    if not chave_api:
        chave_api = os.getenv("GROQ_API_KEY", "")
    if not chave_api:
        raise ValueError(
            "Nenhuma chave de API encontrada. Defina OPENAI_API_KEY (OpenAI/Groq) "
            "ou GROQ_API_KEY no arquivo .env"
        )

    if chave_api.startswith("gsk_"):
        modelo = MODELO_PADRAO
        llm = ChatOpenAI(
            model=modelo,
            temperature=temperatura,
            openai_api_key=chave_api,
            openai_api_base=URL_GROQ,
            timeout=30,
            max_retries=2,
        )
    else:
        if not modelo:
            modelo = "gpt-4o-mini"
        llm = ChatOpenAI(
            model=modelo,
            temperature=temperatura,
            openai_api_key=chave_api,
            timeout=30,
            max_retries=2,
        )
    banco_vetorial = obter_banco_vetorial()
    recuperador = banco_vetorial.as_retriever(search_kwargs={"k": 6})

    cadeia = (
        {
            "context": itemgetter("input") | recuperador | _formatar_documentos,
            "input": itemgetter("input"),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return cadeia, recuperador


def perguntar(
    pergunta: str,
    cadeia=None,
    recuperador=None,
    modelo: str = "",
    temperatura: float = 0.0,
) -> Tuple[str, List[Document]]:
    if cadeia is None or recuperador is None:
        cadeia, recuperador = criar_cadeia_qa(modelo=modelo, temperatura=temperatura)
    resposta = cadeia.invoke({"input": pergunta})
    contexto = recuperador.invoke(pergunta)
    return resposta, contexto


def listar_documentos_disponiveis() -> List[str]:
    """Lista os documentos no diretorio data/documents/ (sem acessar ChromaDB)."""
    dir_docs = os.path.join(os.path.dirname(__file__), "..", "data", "documents")
    if not os.path.isdir(dir_docs):
        return []
    return sorted(
        f for f in os.listdir(dir_docs)
        if os.path.isfile(os.path.join(dir_docs, f)) and not f.startswith(".")
    )


if __name__ == "__main__":
    print("Documentos disponiveis:", listar_documentos_disponiveis())
    print("\n--- Teste do agente ---\n")
    cadeia, recuperador = criar_cadeia_qa()
    testes = [
        "Qual a politica de privacidade do Nubank para dados biometricos?",
        "Como funciona o limite de credito com reserva no Mercado Pago?",
    ]
    for pergunta in testes:
        print(f"Pergunta: {pergunta}")
        resposta, docs = perguntar(pergunta, cadeia=cadeia, recuperador=recuperador)
        print(f"Resposta: {resposta}\n")
