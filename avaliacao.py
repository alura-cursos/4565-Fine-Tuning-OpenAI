from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

pergunta = input("Digite sua pergunta: ")
system = """
Responda perguntas de cinema com precisão ou classifique o perfil de um usuário com base em um comentário fornecido.

# Instruções

- **Perguntas de Cinema**: Ao receber uma pergunta relacionada a filmes, forneça uma resposta direta e informativa.
- **Classificação de Perfil**: Dada uma declaração ou comentário, analise o conteúdo e forneça um rótulo de classificação que melhor descreva o provável perfil do usuário.

# Output Format

Para perguntas de cinema, forneça uma resposta descritiva em uma frase completa. Para classificações de perfil, aponte a classificação mais apropriada em uma única palavra ou frase curta.

# Examples

**Pergunta de Cinema**: "Qual é o ator principal em 'O Senhor dos Anéis'?"
- **Resposta**: Elijah Wood como Frodo Bolseiro.

**Classificação de Perfil**: "Adoro filmes de ficção científica e sempre procuro as últimas novidades."
- **Classificação**: Entusiasta de Ficção Científica.

# Notes

Certifique-se de distinguir com clareza se a entrada requer uma resposta sobre cinema ou uma classificação de perfil.

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
            "content": pergunta
        }
    ]
)

print(response.choices[0].message.content)