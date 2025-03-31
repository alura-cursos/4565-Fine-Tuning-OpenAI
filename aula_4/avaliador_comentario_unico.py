from openai import OpenAI
from dotenv import load_dotenv
import os
import time
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")        

client = OpenAI(api_key=openai_api_key)
response = client.chat.completions.create( 
    model="ft:gpt-4o-2024-08-06:alura-aulas:alura-v3-large:BH0vJyk0", 
    messages=[
        {
            "role": "system", "content": "Você deve avaliar comentários dos usuários, identificando quando se pede sentimentos ou o perfil do usuário. Você só deve tratar de cinema."
        },
        {
            "role": "user", "content": "Comentário: 'Adorei o filme, muito emocionante! O ator principal é incrível e a história me prendeu do começo ao fim. Recomendo para todos os amantes de cinema!'"
        }
    ] 
)

print(response.choices[0].message.content)