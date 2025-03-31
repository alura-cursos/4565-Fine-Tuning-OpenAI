from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os

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

    response.choices[0].message.content

def carrega_csv(caminho_dados):
    try:
        data_frame_cinema = pd.read_csv(caminho_dados)
        return data_frame_cinema
    except Exception as erro:
        print("Não abriu o Data Frame, erro: ", erro)
        return None



load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)