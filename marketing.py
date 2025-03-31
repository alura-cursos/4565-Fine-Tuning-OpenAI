from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import os
import time
from avaliacao import avaliar_perfil_usuario, carrega_csv
    
def combinar_informacoes_filmes(data_frame_cinema, top_n=3):
    contagem_filmes = data_frame_cinema["filme"].value_counts().head(top_n)
    resumo = "Filmes mais comentados: \n"

    for filme, contagem in contagem_filmes.items():
        resumo += f"- {filme}: {contagem} comentários\n"
    return resumo

data_frame_cinema = carrega_csv("dados/comentarios.csv")
print(combinar_informacoes_filmes(data_frame_cinema))

def combinar_perfil_usuario(data_frame_cinema, client):
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

    resumo += "\nPerfis individuais:\n"
    user_ids = data_frame_cinema["user_id"].unique()
    for user_id in user_ids:
        
        comentarios = "\n".join(data_frame_cinema[data_frame_cinema["user_id"] == user_id]["comentario"].dropna().tolist())
        
        perfil = avaliar_perfil_usuario(comentarios, client)
        
        nome = data_frame_cinema[data_frame_cinema["user_id"] == user_id].iloc[0]["nome_completo"]
        resumo += f"- {nome} (ID: {user_id}): {perfil}\n"
    return resumo

def gerar_parecer_marketing(informacoes_filme, informacoes_perfil, client,
                                  model="ft:gpt-4o-2024-08-06:alura-aulas:aluracinemav2:BHEGJ2Xp"):

    prompt = (
        "Baseado nos dados a seguir, sugira uma campanha de marketing criativa "
        "para promover os filmes mais comentados e engajar os usuários com os perfis apresentados.\n\n"
        f"{informacoes_filme}\n"
        f"{informacoes_perfil}\n\n"
        "A sugestão deve considerar as características dos usuários e o potencial dos filmes."
    )

    messages = [
        {
            "role": "system",
            "content": "Você é um consultor de marketing especializado em campanhas criativas."
        },
        {
            "role": "user", 
            "content": prompt
         }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

def main():
    load_dotenv()
    chave_api_openai = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=chave_api_openai)

    data_frame_cinema = carrega_csv("dados/comentarios.csv")
    dados_filmes = combinar_informacoes_filmes(data_frame_cinema=data_frame_cinema)
    dados_perfis = combinar_perfil_usuario(data_frame_cinema, client)

    sugestao_marketing = gerar_parecer_marketing(dados_filmes, dados_perfis, client)

    print("Sugestão: \n")
    print(sugestao_marketing)

if __name__ == "__main__":
    main()