from typing import List

from ferramentas import executar_ferramentas, inicializar_ferramentas


class AgenteConversacional:
    def __init__(self):
        self.historico: List[dict] = []
        inicializar_ferramentas()

    def perguntar(self, mensagem: str) -> str:
        resultados = executar_ferramentas(mensagem)

        textos = []
        todas_fontes = set()

        for resposta, _, fontes in resultados:
            textos.append(resposta)
            for f in fontes:
                todas_fontes.add(f)

        resposta = "\n\n".join(textos)
        if todas_fontes:
            resposta += f"\n\n(Fontes: {', '.join(sorted(todas_fontes))})"

        self.historico.append({"pergunta": mensagem, "resposta": resposta})
        return resposta

    def limpar_historico(self):
        self.historico = []

    def ultimas_perguntas(self, n: int = 5) -> List[dict]:
        return self.historico[-n:]


if __name__ == "__main__":
    agente = AgenteConversacional()
    print("Agente Fintech (Digite 'sair' para encerrar)\n")

    while True:
        pergunta = input("Voce: ")
        if pergunta.lower() in ("sair", "quit", "exit"):
            print("Encerrando.")
            break
        resposta = agente.perguntar(pergunta)
        print(f"\nAgente: {resposta}\n")
