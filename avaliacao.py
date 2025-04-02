from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import time

def avaliar_perfil_usuario(comentarios, client):
    system = """
    Você deve avaliar comentários dos usuários, identificando a qual perfil ele se encaixa.
    Retorne apenas o perfil do usuário, sem explicações adicionais
    """

    response = client.chat.completions.create(
        model= "ft:gpt-4o-2024-08-06:alura-aulas:aluraforum-v1:BHDmI85a",
        messages=[
            {
                "role" : "system",
                "content" : system
            },
            {
                "role" : "user",
                "content": comentarios
            }
        ]
    )

    return response.choices[0].message.content

def carrega_csv(caminho_dados):
    try:
        data_frame_cinema = pd.read_csv(caminho_dados)
        return data_frame_cinema
    except Exception as erro:
        print("Não abriu o Data Frame, erro: ", erro)
        return None

def agrupar_comentarios_por_usuario(data_frame_cinema):
    resultados = []
    grupos = data_frame_cinema.groupby("user_id")

    for user_id, grupo in grupos:
        nome = grupo.iloc[0]["nome_completo"]
        comentarios = "\n".join(grupo["comentario"].dropna().tolist())
        resultados.append(
            {
                "user_id" : user_id,
                "nome" : nome,
                "comentarios" : comentarios
            }
        )

    return resultados

def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)

    data_frame_cinema = carrega_csv("dados/comentarios.csv")
    usuarios = agrupar_comentarios_por_usuario(data_frame_cinema=data_frame_cinema)

    resultado_avaliacao = []
    for usuario in usuarios:
        parecer = avaliar_perfil_usuario(usuario["comentarios"], client)

        resultado_avaliacao.append(
            {
                "user_id" : usuario["user_id"],
                "nome" : usuario["nome"],
                "parecer" : parecer
            }
        )
        print(f"Usuário {usuario["nome"]}")
        print(f"Parecer: {parecer}")
        time.sleep(50)

if __name__ == "__main__":
    main()