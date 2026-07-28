from typing import List

from retriever import criar_cadeia_qa, perguntar


class AgenteConversacional:
    def __init__(self, modelo: str = "gpt-4o-mini", temperatura: float = 0.0):
        cadeia, recuperador = criar_cadeia_qa(modelo=modelo, temperatura=temperatura)
        self.cadeia = cadeia
        self.recuperador = recuperador
        self.historico: List[dict] = []

    def perguntar(self, mensagem: str) -> str:
        resposta, contexto = perguntar(mensagem, cadeia=self.cadeia, recuperador=self.recuperador)
        self.historico.append({"pergunta": mensagem, "resposta": resposta})

        fontes = set()
        for doc in contexto:
            nome = doc.metadata.get("source", "").split("/")[-1]
            if nome:
                fontes.add(nome)

        if fontes:
            resposta += f"\n\n(Fontes: {', '.join(sorted(fontes))})"

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
