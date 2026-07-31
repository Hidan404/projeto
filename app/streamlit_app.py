import hashlib
import json
import os
import sys
from pathlib import Path

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

# Streamlit Sharing: injeta a chave vinda dos secrets nos env vars
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from agent import AgenteConversacional
from retriever import listar_documentos_disponiveis

CHAVE_HISTORICO = "historico_chat_fintech"
LIMITE_HISTORICO = 50

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

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        border-radius: 999px;
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        border: 1px solid #e2e8f0;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.45rem 0.95rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        transition: all 0.15s ease;
        white-space: normal;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #eef2ff, #e0e7ff);
        border-color: #818cf8;
        color: #3730a3;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.15);
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGESTOES = {
    "📊 Banco de dados": [
        "Qual a taxa de juros do credito?",
        "Quais produtos financeiros existem?",
        "Quantos clientes ativos existem?",
        "Qual o faturamento da TechSolucoes?",
    ],
    "📄 Documentos": [
        "Como o Nubank trata meus dados pessoais?",
        "O que a politica de privacidade da Stripe diz sobre cookies?",
    ],
}


@st.cache_resource
def carregar_agente():
    return AgenteConversacional()


@st.cache_data
def _documentos_disponiveis():
    return listar_documentos_disponiveis()


def _historico_do_local_storage():
    """Le o historico persistido no localStorage do navegador."""
    try:
        valor = streamlit_js_eval(
            js_expressions=f"localStorage.getItem('{CHAVE_HISTORICO}')",
            key="ler_historico",
            want_output=True,
        )
        if valor:
            dados = json.loads(valor)
            if isinstance(dados, list):
                return dados
    except Exception:
        pass
    return None


def _salvar_historico(historico):
    """Persiste o historico no localStorage (chave unica por conteudo)."""
    conteudo = json.dumps(historico, ensure_ascii=False)
    chave_hash = hashlib.md5(conteudo.encode()).hexdigest()[:16]
    streamlit_js_eval(
        js_expressions=(
            f"localStorage.setItem('{CHAVE_HISTORICO}', "
            f"{json.dumps(conteudo)})"
        ),
        key=f"salvar_{chave_hash}",
    )


def _limpar_local_storage():
    streamlit_js_eval(
        js_expressions=f"localStorage.removeItem('{CHAVE_HISTORICO}')",
        key="limpar_historico",
    )


if "historico" not in st.session_state:
    st.session_state.historico = []

if "historico_carregado" not in st.session_state:
    historico = _historico_do_local_storage()
    if isinstance(historico, list):
        st.session_state.historico = historico
    st.session_state.historico_carregado = True

documents = _documentos_disponiveis()


def enviar_mensagem(pergunta=None):
    if pergunta is None:
        pergunta = st.session_state.input_usuario.strip()
        st.session_state.input_usuario = ""
    pergunta = pergunta.strip()
    if not pergunta:
        return

    st.session_state.historico.append({"papel": "usuario", "texto": pergunta})

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

    st.session_state.historico = st.session_state.historico[-LIMITE_HISTORICO:]
    _salvar_historico(st.session_state.historico)


for msg in st.session_state.historico:
    with st.chat_message(msg["papel"]):
        st.markdown(msg["texto"])

st.chat_input(
    "Digite sua pergunta...",
    key="input_usuario",
    on_submit=enviar_mensagem,
)


with st.sidebar:
    st.markdown("### ✨ Perguntas sugeridas")
    st.caption("Clique em uma pergunta para enviar")

    i = 0
    for titulo, perguntas in SUGESTOES.items():
        st.markdown(
            f'<p style="margin:0.7rem 0 0.3rem; font-size:0.78rem; '
            f'font-weight:600; color:#64748b; text-transform:uppercase; '
            f'letter-spacing:0.04em;">{titulo}</p>',
            unsafe_allow_html=True,
        )
        for pergunta in perguntas:
            if st.button(pergunta, key=f"sugestao_{i}", use_container_width=True):
                enviar_mensagem(pergunta)
            i += 1

    st.divider()

    st.header("📄 Documentos")
    if documents:
        for doc in documents:
            st.markdown(f"- {doc}")
    else:
        st.info("Nenhum documento encontrado.")

    if st.button("🧹 Limpar histórico", use_container_width=True):
        st.session_state.historico = []
        _limpar_local_storage()
        st.rerun()

    st.divider()
    st.caption(
        "RAG: ChromaDB + all-MiniLM-L6-v2  |  "
        "SQL: SQLite  |  "
        "LLM: Groq Llama 3.3 70B"
    )
