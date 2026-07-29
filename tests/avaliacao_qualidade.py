"""Avaliacao de qualidade do RAG.

Para cada documento indexado, faz uma serie de perguntas e avalia
se a resposta foi satisfatoria, se citou a fonte correta e se
reconheceu quando nao sabia a resposta.
"""

import os
import sys
import time

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

from retriever import criar_cadeia_qa, perguntar, listar_documentos_disponiveis


# Perguntas especificas por documento
PERGUNTAS = {
    "nubank": [
        "Quais dados pessoais o Nubank coleta?",
        "Qual a finalidade dos dados biometricos no Nubank?",
        "Com quem o Nubank compartilha dados?",
        "Como solicitar a exclusao de dados no Nubank?",
        "Quais sao os direitos do titular segundo o Nubank?",
        "O Nubank utiliza cookies? Para que?",
        "Como o Nubank trata dados de criancas?",
        "Por quanto tempo o Nubank retem dados pessoais?",
        "O Nubank realiza transferencia internacional de dados?",
        "Como o Nubank protege os dados contra vazamentos?",
    ],
    "stripe": [
        "Como a Stripe trata dados pessoais?",
        "A Stripe vende dados de usuarios?",
        "Quais direitos o titular tem sobre seus dados na Stripe?",
        "Como solicitar a exclusao de dados na Stripe?",
        "A Stripe utiliza cookies?",
        "Quais medidas de seguranca a Stripe adota?",
        "Como a Stripe trata transferencia internacional de dados?",
        "Por quanto tempo a Stripe mantem dados?",
        "Quais dados a Stripe coleta automaticamente?",
        "Como a Stripe notifica sobre mudancas na politica?",
    ],
    "mercadopago": [
        "Quais as taxas do Mercado Pago para cartao de credito?",
        "Como funciona o limite de credito com reserva?",
        "Quais documentos sao necessarios para contratar?",
        "Qual o prazo de pagamento da fatura?",
        "Como cancelar o contrato do cartao?",
        "Quais os juros do rotativo no Mercado Pago?",
        "Como funciona a portabilidade de saldo?",
        "Quais as obrigacoes do contratante?",
        "O que acontece em caso de atraso no pagamento?",
        "Como solicitar aumento de limite?",
    ],
    "picpay": [
        "Quais dados pessoais o PicPay coleta?",
        "Como o PicPay trata dados biometricos?",
        "Com quem o PicPay compartilha dados?",
        "Quais os direitos dos usuarios segundo o PicPay?",
        "Como excluir a conta no PicPay?",
        "O PicPay utiliza cookies?",
        "Por quanto tempo o PicPay guarda dados?",
        "O PicPay realiza transferencia internacional de dados?",
        "Quais medidas de seguranca o PicPay adota?",
        "Como o PicPay trata dados de menores?",
    ],
}

PERGUNTAS_GERAIS = [
    "Qual a previsao do tempo para amanha?",
    "Quem ganhou a copa do mundo de 2022?",
    "Me explique como fazer um bolo de chocolate",
    "Qual o valor do bitcoin hoje?",
]


def normalizar(texto: str) -> str:
    """Remove acentos para comparacao."""
    return (
        texto.lower()
        .replace("ã", "a")
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ú", "u")
        .replace("ç", "c")
    )


def avaliar_pergunta(pergunta: str, cadeia, recuperador, documento_esperado: str = "") -> dict:
    """Avalia uma pergunta e retorna metricas."""
    inicio = time.time()
    resposta, docs_recuperados = perguntar(pergunta, cadeia=cadeia, recuperador=recuperador)
    duracao = time.time() - inicio

    fontes_encontradas = set()
    for doc in docs_recuperados:
        nome = doc.metadata.get("source", "").split("/")[-1]
        if nome:
            fontes_encontradas.add(nome.lower())

    resp_norm = normalizar(resposta)

    # Metricas
    tem_resposta = len(resposta.strip()) > 20
    citou_fonte_correta = (
        any(documento_esperado.lower() in f for f in fontes_encontradas)
        if documento_esperado
        else True
    )
    reconheceu_desconhecimento = "nao encontrei" in resp_norm or "nao encontrada" in resp_norm
    respondeu_com_documento = bool(fontes_encontradas)

    return {
        "pergunta": pergunta,
        "resposta": resposta[:200],
        "duracao_seg": round(duracao, 1),
        "tem_resposta": tem_resposta,
        "citou_fonte_correta": citou_fonte_correta,
        "reconheceu_desconhecimento": reconheceu_desconhecimento,
        "respondeu_com_documento": respondeu_com_documento,
        "fontes": sorted(fontes_encontradas),
    }


def executar_avaliacao():
    """Executa a avaliacao completa."""
    print("=" * 60)
    print("AVALIACAO DE QUALIDADE DO RAG")
    print("=" * 60)

    cadeia, recuperador = criar_cadeia_qa()
    documentos = listar_documentos_disponiveis()

    print(f"\nDocumentos indexados: {len(documentos)}")
    for d in documentos:
        print(f"  - {d}")

    resultados = {
        "documentos": {},
        "fora_de_contexto": [],
        "totais": {"total": 0, "sucesso": 0, "falha": 0},
    }

    # Testar perguntas especificas por documento
    for palavra_chave, perguntas in PERGUNTAS.items():
        # Encontrar qual documento corresponde
        doc_correspondente = next(
            (d for d in documentos if palavra_chave in d.lower()), None
        )
        print(f"\n--- Documento: {doc_correspondente or palavra_chave} ---")

        resultados_doc = []
        for pergunta in perguntas:
            resultado = avaliar_pergunta(pergunta, cadeia, recuperador, doc_correspondente or "")
            resultados_doc.append(resultado)

            status = (
                "OK" if resultado["tem_resposta"] and resultado["citou_fonte_correta"]
                else "FALHA"
            )
            print(f"  [{status}] {pergunta[:60]:<60s} ({resultado['duracao_seg']}s)")

        resultados["documentos"][palavra_chave] = resultados_doc

    # Testar perguntas fora de contexto
    print("\n--- Perguntas fora de contexto ---")
    for pergunta in PERGUNTAS_GERAIS:
        resultado = avaliar_pergunta(pergunta, cadeia, recuperador)
        resultados["fora_de_contexto"].append(resultado)

        status = "OK" if resultado["reconheceu_desconhecimento"] else "FALHA"
        print(f"  [{status}] {pergunta[:60]:<60s} ({resultado['duracao_seg']}s)")

    # Calcular metricas agregadas
    total_perguntas = 0
    acertos = 0

    for doc_name, res_list in resultados["documentos"].items():
        for r in res_list:
            total_perguntas += 1
            if r["tem_resposta"] and r["citou_fonte_correta"]:
                acertos += 1

    for r in resultados["fora_de_contexto"]:
        total_perguntas += 1
        if r["reconheceu_desconhecimento"]:
            acertos += 1

    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS")
    print("=" * 60)
    print(f"Total de perguntas: {total_perguntas}")
    print(f"Acertos: {acertos}")
    print(f"Acurácia: {acertos / total_perguntas * 100:.1f}%")
    print(f"Tempo medio: {sum(r['duracao_seg'] for doc in resultados['documentos'].values() for r in doc) / max(len([r for doc in resultados['documentos'].values() for r in doc]), 1):.1f}s")

    return resultados


if __name__ == "__main__":
    executar_avaliacao()
