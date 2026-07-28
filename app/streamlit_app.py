import sys
from pathlib import Path

import streamlit as st

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
    "Assistente virtual especializado em documentos internos da fintech. "
    "Faça perguntas sobre políticas, contratos e procedimentos."
)


@st.cache_resource
def carregar_agente():
    return AgenteConversacional()


@st.cache_data
def carregar_documentos():
    try:
        return listar_documentos_disponiveis()
    except Exception:
        return []


if "historico" not in st.session_state:
    st.session_state.historico = []


def enviar_mensagem():
    pergunta = st.session_state.input_usuario.strip()
    if not pergunta:
        return

    st.session_state.historico.append({"papel": "usuario", "texto": pergunta})
    st.session_state.input_usuario = ""

    with st.spinner("Consultando documentos..."):
        try:
            agente = carregar_agente()
            resposta = agente.perguntar(pergunta)
            st.session_state.historico.append({"papel": "agente", "texto": resposta})
        except Exception as e:
            st.session_state.historico.append(
                {"papel": "agente", "texto": f"**Erro:** {e}"}
            )


for msg in st.session_state.historico:
    with st.chat_message(msg["papel"]):
        st.markdown(msg["texto"])

st.chat_input(
    "Digite sua pergunta sobre os documentos...",
    key="input_usuario",
    on_submit=enviar_mensagem,
)


with st.sidebar:
    st.header("📄 Documentos Disponíveis")
    docs = carregar_documentos()
    if docs:
        for doc in docs:
            st.markdown(f"- {doc}")
    else:
        st.info("Nenhum documento encontrado no banco vetorial.")

    st.divider()

    if st.button("🧹 Limpar histórico", use_container_width=True):
        st.session_state.historico = []
        st.rerun()

    st.divider()
    st.caption(
        "Modelo: Groq Llama 3.3 70B  |  "
        "Embeddings: all-MiniLM-L6-v2  |  "
        "Banco: ChromaDB"
    )
