import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def load_csv(data_path):
    """
    Carrega os dados do CSV em um DataFrame do pandas.

    :param data_path: Caminho para o arquivo CSV.
    :return: DataFrame com os dados ou None em caso de erro.
    """
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as error:
        print(f"Erro ao ler o CSV: {error}")
        return None


def aggregate_movie_info(df, top_n=3):
    """
    Agrega informações dos filmes mais comentados.

    :param df: DataFrame com os dados.
    :param top_n: Número de filmes mais comentados a retornar.
    :return: String resumo dos filmes mais comentados.
    """
    movie_counts = df["filme"].value_counts().head(top_n)
    resumo = "Filmes mais comentados:\n"
    for filme, contagem in movie_counts.items():
        resumo += f"- {filme}: {contagem} comentários\n"
    return resumo


def aggregate_user_profiles(df):
    """
    Agrega informações sobre os perfis dos usuários.

    :param df: DataFrame com os dados.
    :return: String resumo dos perfis dos usuários.
    """
    total_usuarios = df["user_id"].nunique()
    genero_counts = df["genero"].value_counts()
    sentimento_counts = df["sentimento"].value_counts()
    idade_media = df["idade"].mean()
    top_localizacoes = df["localizacao"].value_counts().head(3)

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


def generate_marketing_suggestion(movie_info, user_profiles, client,
                                  model="ft:gpt-4o-2024-08-06:alura-aulas:alura-v3-large:BH0vJyk0"):
    """
    Gera uma sugestão para o time de marketing com base nos dados agregados.

    :param movie_info: String com informações dos filmes.
    :param user_profiles: String com informações dos perfis dos usuários.
    :param client: Instância do OpenAI configurada com a API key.
    :param model: Modelo a ser utilizado na API.
    :return: Sugestão gerada pelo modelo.
    """
    prompt = (
        "Baseado nos dados a seguir, sugira uma campanha de marketing criativa "
        "para promover os filmes mais comentados e engajar os usuários com os perfis apresentados.\n\n"
        f"{movie_info}\n"
        f"{user_profiles}\n\n"
        "A sugestão deve considerar as características dos usuários e o potencial dos filmes."
    )

    messages = [
        {
            "role": "system",
            "content": "Você é um consultor de marketing especializado em campanhas criativas."
        },
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("A variável OPENAI_API_KEY não está definida no .env")
        return

    client = OpenAI(api_key=openai_api_key)

    # Atualize com o caminho do seu arquivo CSV
    data_path = "aula_5/comentarios.csv"
    df = load_csv(data_path)
    if df is None:
        return

    movie_info = aggregate_movie_info(df)
    user_profiles = aggregate_user_profiles(df)

    suggestion = generate_marketing_suggestion(movie_info, user_profiles, client)
    print("Sugestão para o time de marketing:")
    print(suggestion)
    # Opcional: aguardar um pouco para evitar chamadas seguidas à API
    time.sleep(1)


if __name__ == "__main__":
    main()
