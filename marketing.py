from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import time

def carrega_csv(caminho_dados):
    try:
        data_frame_cinema = pd.read_csv(caminho_dados)
        return data_frame_cinema
    except Exception as erro:
        print("Não abriu o Data Frame, erro: ", erro)
        return None
    
def combinar_informacoes_filmes(data_frame_cinema, top_n=3):
    contagem_filmes = data_frame_cinema["filme"].value_counts().head(top_n)
    resumo = "Filmes mais comentados: \n"

    for filme, contagem in contagem_filmes.items():
        resumo += f"- {filme}: {contagem} comentários\n"
    return resumo

data_frame_cinema = carrega_csv("dados/comentarios.csv")
print(combinar_informacoes_filmes(data_frame_cinema))

def combinar_perfil_usuario(data_frame_cinema):
    total_usuarios = data_frame_cinema["user_id"].nunique()
    genero_counts = data_frame_cinema["genero"].value_counts()
    sentimento_counts = data_frame_cinema["sentimento"].value_counts()
    idade_media = data_frame_cinema["idade"].mean()
    top_localizacoes = data_frame_cinema["localizacao"].value_counts().head(3)

    resumo = f"Perfis dos usuários (total: {total_usuarios}):\n"
    resumo += "Gênero:\n"
    for genero, contagem in genero_counts.items():
        resumo += f"- {genero}: {contagem}\n"
    resumo += "Sentimento dos comentários:\n"
    for sentimento, contagem in sentimento_counts.items():
        resumo += f"- {sentimento}: {contagem}\n"
    resumo += f"Idade média: {idade_media:.1f}\n"
    resumo += "Principais localizações:\n"
    for local, contagem in top_localizacoes.items():
        resumo += f"- {local}: {contagem}\n"
    return resumo