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

load_dotenv()

URL_GROQ = "https://api.groq.com/openai/v1"
MODELO_PADRAO = "llama-3.3-70b-versatile"

PROMPT_SISTEMA = """
Voce e um assistente virtual especializado em documentos internos de uma fintech.
Use APENAS o contexto fornecido abaixo para responder a pergunta do colaborador.

Se a resposta nao estiver claramente no contexto, diga:
"Nao encontrei essa informacao nos documentos disponiveis."

Sempre mencione ao final o nome do documento de onde a informacao foi extraida.

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
        )
    else:
        if not modelo:
            modelo = "gpt-4o-mini"
        llm = ChatOpenAI(model=modelo, temperature=temperatura, openai_api_key=chave_api)
    banco_vetorial = obter_banco_vetorial()
    recuperador = banco_vetorial.as_retriever(search_kwargs={"k": 4})

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
    banco_vetorial = obter_banco_vetorial()
    resultados = banco_vetorial.similarity_search("", k=100)
    fontes = set()
    for doc in resultados:
        fonte = doc.metadata.get("source", "")
        if fonte:
            fontes.add(fonte.split("/")[-1])
    return sorted(fontes)


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
