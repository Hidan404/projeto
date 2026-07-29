import csv
import os
import sqlite3
from typing import Any, Dict, List, Optional

DIR_DADOS = os.path.join(os.path.dirname(__file__), "..", "data")
CAMINHO_BANCO = os.path.join(DIR_DADOS, "fintech.db")


def inicializar_banco() -> str:
    """Cria as tabelas e importa os CSVs se o banco ainda nao existir."""
    if os.path.exists(CAMINHO_BANCO):
        return CAMINHO_BANCO

    conn = sqlite3.connect(CAMINHO_BANCO)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY,
            nome_empresa TEXT NOT NULL,
            cnpj TEXT UNIQUE NOT NULL,
            segmento TEXT,
            data_cadastro TEXT,
            status TEXT CHECK(status IN ('ativo','inativo')),
            plano TEXT,
            faturamento_mensal_estimado REAL
        );

        CREATE TABLE IF NOT EXISTS produtos_financeiros (
            id INTEGER PRIMARY KEY,
            nome TEXT NOT NULL,
            tipo TEXT CHECK(tipo IN ('credito','servico')),
            descricao TEXT,
            taxa_juros_mensal REAL,
            prazo_minimo_dias INTEGER,
            prazo_maximo_dias INTEGER,
            valor_minimo REAL,
            valor_maximo REAL,
            requisitos TEXT
        );

        CREATE TABLE IF NOT EXISTS transacoes (
            id INTEGER PRIMARY KEY,
            cliente_id INTEGER NOT NULL,
            tipo TEXT CHECK(tipo IN ('entrada','saida')),
            valor REAL NOT NULL,
            data TEXT NOT NULL,
            status TEXT CHECK(status IN ('concluido','pendente','cancelado')),
            produto_id INTEGER,
            descricao TEXT,
            FOREIGN KEY (cliente_id) REFERENCES clientes(id),
            FOREIGN KEY (produto_id) REFERENCES produtos_financeiros(id)
        );
    """)

    _importar_csv(conn, "clientes.csv")
    _importar_csv(conn, "produtos_financeiros.csv")
    _importar_csv(conn, "transacoes.csv")

    conn.commit()
    conn.close()
    return CAMINHO_BANCO


def _importar_csv(conn: sqlite3.Connection, nome_arquivo: str):
    caminho = os.path.join(DIR_DADOS, nome_arquivo)
    if not os.path.exists(caminho):
        return
    tabela = nome_arquivo.replace(".csv", "")
    with open(caminho, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        colunas = list(leitor.fieldnames)
        placeholders = ",".join("?" for _ in colunas)
        cols_str = ",".join(colunas)
        for linha in leitor:
            valores = [linha[c] for c in colunas]
            conn.execute(
                f"INSERT OR IGNORE INTO {tabela} ({cols_str}) VALUES ({placeholders})",
                valores,
            )


def executar_sql(sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """Executa uma consulta SQL e retorna lista de dicionarios."""
    conn = sqlite3.connect(CAMINHO_BANCO)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(sql, params or ())
        resultados = [dict(linha) for linha in cursor.fetchall()]
        return resultados
    finally:
        conn.close()


def listar_tabelas() -> List[str]:
    conn = sqlite3.connect(CAMINHO_BANCO)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tabelas = [linha[0] for linha in cursor.fetchall()]
    conn.close()
    return tabelas


def descrever_tabela(tabela: str) -> List[Dict[str, str]]:
    conn = sqlite3.connect(CAMINHO_BANCO)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({tabela})")
    colunas = [
        {"nome": linha[1], "tipo": linha[2], "nulo": not linha[3]}
        for linha in cursor.fetchall()
    ]
    conn.close()
    return colunas


def resumo_banco() -> str:
    """Retorna uma descricao textual do schema para enviar ao LLM."""
    linhas = ["BANCO DE DADOS FINTECH (SQLite)", "=" * 40]
    for tabela in listar_tabelas():
        qtd = executar_sql(f"SELECT COUNT(*) as total FROM {tabela}")[0]["total"]
        linhas.append(f"\nTabela: {tabela} ({qtd} registros)")
        for col in descrever_tabela(tabela):
            linhas.append(f"  - {col['nome']} ({col['tipo']})")
    return "\n".join(linhas)


def perguntar_banco(pergunta: str, linhas_max: int = 15) -> str:
    """Interpreta a pergunta em linguagem natural e retorna resposta formatada.

    Usa um conjunto de queries pre-definidas para questoes comuns.
    Se nenhuma query casar, retorna mensagem informativa.
    """
    p = pergunta.lower()

    # --- CLIENTES ---
    if "cliente" in p or "empresa" in p or "cnpj" in p or "cadastro" in p:
        if "todos" in p or "listar" in p or "quais" in p or "quantos" in p:
            if "ativo" in p:
                resultado = executar_sql(
                    "SELECT id, nome_empresa, segmento, plano, faturamento_mensal_estimado "
                    "FROM clientes WHERE status='ativo' ORDER BY faturamento_mensal_estimado DESC"
                )
                prefixo = "Clientes ativos"
            elif "inativo" in p:
                resultado = executar_sql(
                    "SELECT id, nome_empresa, segmento, status FROM clientes WHERE status='inativo'"
                )
                prefixo = "Clientes inativos"
            else:
                resultado = executar_sql(
                    "SELECT id, nome_empresa, segmento, status, plano FROM clientes ORDER BY id"
                )
                prefixo = "Todos os clientes"
        elif "maior" in p or "mais" in p or "top" in p:
            resultado = executar_sql(
                "SELECT nome_empresa, segmento, faturamento_mensal_estimado "
                "FROM clientes WHERE status='ativo' ORDER BY faturamento_mensal_estimado DESC LIMIT 5"
            )
            prefixo = "Top 5 clientes por faturamento"
        elif "media" in p or "medio" in p or "media" in p:
            resultado = executar_sql(
                "SELECT COUNT(*) as total, AVG(faturamento_mensal_estimado) as media "
                "FROM clientes WHERE status='ativo'"
            )
            prefixo = "Media de faturamento"
        elif "segment" in p:
            resultado = executar_sql(
                "SELECT segmento, COUNT(*) as total FROM clientes GROUP BY segmento ORDER BY total DESC"
            )
            prefixo = "Clientes por segmento"
        elif "plano" in p:
            resultado = executar_sql(
                "SELECT plano, COUNT(*) as total FROM clientes GROUP BY plano ORDER BY total DESC"
            )
            prefixo = "Clientes por plano"
        else:
            resultado = executar_sql(
                "SELECT id, nome_empresa, segmento, status, plano, faturamento_mensal_estimado "
                "FROM clientes ORDER BY id LIMIT ?", (linhas_max,)
            )
            prefixo = "Clientes cadastrados"
        return _formatar_resultado(prefixo, resultado)

    # --- PRODUTOS ---
    if "produto" in p or "servi" in p or "credito" in p or "cartao" in p or "linha" in p:
        if "todos" in p or "listar" in p or "quais" in p or "dispon" in p:
            resultado = executar_sql(
                "SELECT id, nome, tipo, descricao, taxa_juros_mensal, "
                "valor_minimo, valor_maximo, requisitos FROM produtos_financeiros ORDER BY id"
            )
            prefixo = "Produtos financeiros disponiveis"
        elif "credito" in p or "emprestimo" in p or "giro" in p:
            resultado = executar_sql(
                "SELECT id, nome, tipo, descricao, taxa_juros_mensal, valor_minimo, valor_maximo "
                "FROM produtos_financeiros WHERE tipo='credito' ORDER BY taxa_juros_mensal"
            )
            prefixo = "Produtos de credito"
        elif "servico" in p or "servi" in p:
            resultado = executar_sql(
                "SELECT id, nome, tipo, descricao FROM produtos_financeiros WHERE tipo='servico'"
            )
            prefixo = "Servicos financeiros"
        elif "menor" in p or "barato" in p or "baixo" in p:
            resultado = executar_sql(
                "SELECT nome, tipo, taxa_juros_mensal FROM produtos_financeiros "
                "WHERE tipo='credito' ORDER BY taxa_juros_mensal LIMIT 3"
            )
            prefixo = "Produtos com menores taxas"
        else:
            resultado = executar_sql(
                "SELECT id, nome, tipo, descricao, taxa_juros_mensal FROM produtos_financeiros LIMIT ?",
                (linhas_max,),
            )
            prefixo = "Produtos financeiros"
        return _formatar_resultado(prefixo, resultado)

    # --- TRANSACOES ---
    if "transac" in p or "moviment" in p or "entrada" in p or "saida" in p or "receita" in p:
        if "todas" in p or "todas as" in p or "listar" in p:
            resultado = executar_sql(
                "SELECT t.id, c.nome_empresa, t.tipo, t.valor, t.data, t.status, t.descricao "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "ORDER BY t.data DESC LIMIT ?", (linhas_max,)
            )
            prefixo = "Transacoes recentes"
        elif "pendente" in p:
            resultado = executar_sql(
                "SELECT t.id, c.nome_empresa, t.tipo, t.valor, t.data, t.descricao "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "WHERE t.status='pendente' ORDER BY t.data"
            )
            prefixo = "Transacoes pendentes"
        elif "total" in p:
            if "entrada" in p or "receita" in p:
                resultado = executar_sql(
                    "SELECT SUM(valor) as total FROM transacoes WHERE tipo='entrada' AND status='concluido'"
                )
                prefixo = "Total de entradas (concluidas)"
            elif "saida" in p or "despesa" in p:
                resultado = executar_sql(
                    "SELECT SUM(valor) as total FROM transacoes WHERE tipo='saida' AND status='concluido'"
                )
                prefixo = "Total de saidas (concluidas)"
            else:
                resultado = executar_sql(
                    "SELECT SUM(CASE WHEN tipo='entrada' THEN valor ELSE 0 END) as total_entradas, "
                    "SUM(CASE WHEN tipo='saida' THEN valor ELSE 0 END) as total_saidas "
                    "FROM transacoes WHERE status='concluido'"
                )
                prefixo = "Resumo financeiro"
        elif "conclu" in p or "realizada" in p:
            resultado = executar_sql(
                "SELECT t.id, c.nome_empresa, t.tipo, t.valor, t.data, t.descricao "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "WHERE t.status='concluido' ORDER BY t.data DESC LIMIT ?", (linhas_max,)
            )
            prefixo = "Transacoes concluidas"
        elif "cancel" in p:
            resultado = executar_sql(
                "SELECT t.id, c.nome_empresa, t.tipo, t.valor, t.data, t.descricao "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "WHERE t.status='cancelado'"
            )
            prefixo = "Transacoes canceladas"
        elif "maior" in p or "mais valiosa" in p:
            resultado = executar_sql(
                "SELECT c.nome_empresa, t.valor, t.data, t.descricao "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "WHERE t.status='concluido' ORDER BY t.valor DESC LIMIT 5"
            )
            prefixo = "Maiores transacoes"
        else:
            resultado = executar_sql(
                "SELECT t.id, c.nome_empresa, t.tipo, t.valor, t.data, t.status "
                "FROM transacoes t JOIN clientes c ON t.cliente_id = c.id "
                "ORDER BY t.data DESC LIMIT ?", (linhas_max,)
            )
            prefixo = "Transacoes"
        return _formatar_resultado(prefixo, resultado)

    # --- SQL GENERICO ---
    palavras = p.split()
    for palavra in palavras:
        try:
            if palavra.isdigit() and len(palavra) <= 3:
                resultado = executar_sql(
                    "SELECT id, nome_empresa, segmento, status, plano FROM clientes WHERE id = ?",
                    (int(palavra),),
                )
                if resultado:
                    return _formatar_resultado(f"Cliente #{palavra}", resultado)
        except ValueError:
            pass

    return "Nao consegui interpretar sua pergunta sobre o banco de dados. Tente perguntar sobre clientes, produtos financeiros ou transacoes."


def _formatar_resultado(titulo: str, resultados: List[Dict]) -> str:
    if not resultados:
        return f"{titulo}: (nenhum resultado encontrado)"

    cabecalho = f"📊 {titulo}\n" + "-" * 40
    linhas = []

    for r in resultados:
        partes = [f"{k}: {v}" for k, v in r.items() if v is not None]
        linhas.append(" | ".join(partes))

    return cabecalho + "\n" + "\n".join(linhas)


if __name__ == "__main__":
    inicializar_banco()
    print(resumo_banco())
    print("\n" + "=" * 50)
    print(perguntar_banco("Quais clientes ativos temos?"))
    print("\n" + "=" * 50)
    print(perguntar_banco("Listar produtos de credito"))
    print("\n" + "=" * 50)
    print(perguntar_banco("Qual o total de entradas?"))
