import os
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


def avaliar_perfil_usuario(comentarios, client):
    mensagens = [
        {
            "role": "system",
            "content": ("Você deve avaliar comentários dos usuários, identificando a qual perfil ele se encaixa."
                        "Retorne apenas o perfil do usuário, sem explicações adicionais.")
        },
        {"role": "user", "content": f"Comentários: '{comentarios}'"}
    ]
    
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:alura-aulas:alura-v3-large:BH0vJyk0",
        messages=mensagens
    )
    
    return response.choices[0].message.content


def carregar_csv(data_path):
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as error:
        print("Erro ao ler o CSV:", error)
        return None


def agrupar_comentarios_por_usuario(df):
    resultados = []
    grupos = df.groupby("user_id")
    for user_id, grupo in grupos:
        nome = grupo.iloc[0]["nome_completo"]
        # Junta todos os comentários do usuário em uma única string
        comentarios = "\n".join(grupo["comentario"].dropna().tolist())
        resultados.append({"user_id": user_id, "nome": nome, "comentarios": comentarios})
    return resultados


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("A variável OPENAI_API_KEY não está definida no .env")
        return

    client = OpenAI(api_key=openai_api_key)

    data_path = "aula_4/comentarios.csv"  # Atualize com o caminho do seu CSV
    df = carregar_csv(data_path)
    if df is None:
        return

    usuarios = agrupar_comentarios_por_usuario(df)

    resultados_avaliacao = []
    for usuario in usuarios:
        parecer = avaliar_perfil_usuario(usuario["comentarios"], client)
        resultados_avaliacao.append({
            "user_id": usuario["user_id"],
            "nome": usuario["nome"],
            "parecer": parecer
        })
        print(f"Usuário: {usuario['nome']} (ID: {usuario['user_id']})")
        print("Comentário:", parecer)
        print("=" * 50)
        time.sleep(1)  # Aguarda um pouco para evitar sobrecarga na API

    # Opcional: salvar os resultados em um arquivo CSV
    df_resultados = pd.DataFrame(resultados_avaliacao)
    df_resultados.to_csv("parecer_perfis_usuarios.csv", index=False)
    print("Resultados salvos em 'parecer_perfis_usuarios.csv'")


if __name__ == "__main__":
    main()
