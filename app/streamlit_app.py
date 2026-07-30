import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit Sharing: injeta a chave vinda dos secrets nos env vars
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from agent import AgenteConversacional
from retriever import listar_documentos_disponiveis

st.set_page_config(
    page_title="Agente Fintech",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agente Fintech")
st.markdown(
    "Assistente virtual que consulta **documentos internos** "
    "e **banco de dados** da fintech para responder perguntas "
    "de colaboradores."
)


@st.cache_resource
def carregar_agente():
    return AgenteConversacional()


@st.cache_data
def _documentos_disponiveis():
    return listar_documentos_disponiveis()

documents = _documentos_disponiveis()

if "historico" not in st.session_state:
    st.session_state.historico = []


def enviar_mensagem():
    pergunta = st.session_state.input_usuario.strip()
    if not pergunta:
        return

    st.session_state.historico.append({"papel": "usuario", "texto": pergunta})
    st.session_state.input_usuario = ""

    with st.spinner("Consultando documentos e banco de dados..."):
        try:
            agente = carregar_agente()
            resposta = agente.perguntar(pergunta)
            st.session_state.historico.append({"papel": "agente", "texto": resposta})
        except TimeoutError as e:
            st.session_state.historico.append(
                {"papel": "agente", "texto": "**Servico temporariamente indisponivel.** A API de IA demorou muito para responder. Tente novamente em alguns instantes."}
            )
        except Exception as e:
            st.session_state.historico.append(
                {"papel": "agente", "texto": f"**Erro inesperado:** {e}"}
            )

    st.session_state.historico = st.session_state.historico[-50:]

for msg in st.session_state.historico:
    with st.chat_message(msg["papel"]):
        st.markdown(msg["texto"])

st.chat_input(
    "Digite sua pergunta...",
    key="input_usuario",
    on_submit=enviar_mensagem,
)


with st.sidebar:
    st.header("📄 Documentos")
    if documents:
        for doc in documents:
            st.markdown(f"- {doc}")
    else:
        st.info("Nenhum documento encontrado.")

    if st.button("🧹 Limpar histórico", use_container_width=True):
        st.session_state.historico = []
        st.rerun()

    st.divider()
    st.caption(
        "RAG: ChromaDB + all-MiniLM-L6-v2  |  "
        "SQL: SQLite  |  "
        "LLM: Groq Llama 3.3 70B"
    )
