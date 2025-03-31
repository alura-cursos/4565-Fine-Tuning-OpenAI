import pandas as pd
import json

def show_as_json(df, index):
    """
    Mostra uma linha do DataFrame como JSON.
    :param df: DataFrame
    :param index: índice da linha a ser mostrada
    """
    # Seleciona a linha com o índice fornecido
    linha = df.iloc[index]

    # Converte a linha para um dicionário
    linha_dict = linha.to_dict()

    # Converte o dicionário para uma string JSON com formatação bonita
    linha_json = json.dumps(linha_dict, indent=4, ensure_ascii=False)

    print(linha_json)

# Carregar o DataFrame  
df = pd.read_csv('dados/comentarios.csv', sep=',', encoding='utf-8')

# Exibir as primeiras linhas de cada usuário em formato JSON
# for usuario in df['username'].unique():
#     print(f'Usuário: {usuario}')
#     show_as_json(df[df['username'] == usuario], 0)
#     print()

# exibir primeiro comentário em que a usuario_respondido é diferente de NaN
df_comentarios = df[df['usuario_respondido'].notna()]
print(f'Primeiro comentário em que a postagem é diferente de NaN:')     
show_as_json(df_comentarios, 0)
show_as_json(df, 39)