from typing import List

from ferramentas import executar_ferramentas


class AgenteConversacional:
    def __init__(self):
        self.historico: List[dict] = []

    def perguntar(self, mensagem: str) -> str:
        resultados = executar_ferramentas(mensagem)

        textos = []
        todos_fontes = set()

        for resposta, nome_ferramenta, fontes in resultados:
            textos.append(resposta)
            for f in fontes:
                todos_fontes.add(f)

        if len(textos) == 1:
            resposta = textos[0]
        else:
            resposta = (
                "Combinando informacoes dos documentos e do banco de dados:\n\n"
                + "\n\n---\n\n".join(textos)
            )

        if todos_fontes:
            resposta += f"\n\n(Fontes: {', '.join(sorted(todos_fontes))})"

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
