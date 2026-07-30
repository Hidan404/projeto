# Agente Fintech

Assistente virtual que responde perguntas de colaboradores sobre documentos internos (privacidade, contratos) e dados estruturados (clientes, produtos, transacoes) de uma fintech.

## Arquitetura

```
Usuario → Streamlit UI → Agente Conversacional
                            ├── Classificador (pergunta → ferramenta)
                            ├── RAG (ChromaDB + HuggingFace Embeddings + Groq LLM)
                            └── SQL (SQLite em memoria com queries pre-definidas)
```

- **RAG**: ChromaDB em memoria com `paraphrase-multilingual-MiniLM-L12-v2`
- **LLM**: Groq (`llama-3.3-70b-versatile`) via API compativel com OpenAI
- **SQLite**: Banco em memoria com dados de 10 clientes, 6 produtos, 22 transacoes
- **UI**: Streamlit com cache e historico limitado a 50 mensagens

## Requisitos

- Python 3.11+
- Chave de API [Groq](https://console.groq.com/keys)

## Setup Local

```bash
# Clone o repositorio
git clone <repo-url>
cd projeto

# Crie e ative um virtualenv
python3 -m venv .venv
source .venv/bin/activate

# Instale as dependencias
pip install -r requirements.txt

# Configure a chave Groq
cp .env.example .env
# Edite .env e coloque sua chave: GROQ_API_KEY="gsk_sua-chave-aqui"

# Execute
streamlit run app/streamlit_app.py
```

## Deploy no Streamlit Sharing

1. Faca push do codigo para um repositorio GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io) e conecte o repositorio
3. Em **Settings → Secrets**, adicione:
   ```
   GROQ_API_KEY = "gsk_sua-chave-aqui"
   ```
4. Clique em **Deploy**

O app usa ChromaDB e SQLite em memoria — nenhum dado persistente e necessario.

## Estrutura

```
projeto/
  app/streamlit_app.py    # Interface do usuario
  src/
    agent.py              # Agente conversacional
    ferramentas.py        # Classificador e execucao de ferramentas
    retriever.py          # Cadeia RAG (QA com Groq)
    vector_store.py       # ChromaDB em memoria e indexacao
    banco.py              # SQLite em memoria com queries
    loader.py             # Carregamento de documentos (PDF, DOCX, etc.)
    chunker.py            # Divisao em chunks (1500/300)
  data/documents/         # Documentos fonte (PDF, MD, TXT)
  tests/                  # Testes unitarios e de integracao
```
