"""
Módulo para análise e validação de datasets para fine-tuning de chat models.

Contém funções para:
- Carregar datasets em formato JSONL;
- Validar o formato dos registros;
- Calcular estatísticas de tokens;
- Estimar número de épocas e custo do treinamento.
"""

import json
import tiktoken
import numpy as np
from collections import defaultdict


def load_dataset(data_path):
    """
    Carrega um dataset a partir de um arquivo JSONL.
    
    Cada linha deve conter um objeto JSON válido.
    
    :param data_path: Caminho do arquivo JSONL.
    :return: Lista de registros (objetos JSON/dicionários).
    """
    dataset = []
    try:
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        dataset.append(record)
                    except json.JSONDecodeError as error:
                        print("Erro de decodificação:", error)
    except Exception as error:
        print("Erro ao ler o arquivo:", error)
    return dataset


def print_initial_stats(dataset):
    """
    Exibe estatísticas iniciais do dataset.
    
    :param dataset: Lista de registros.
    """
    print("Num examples:", len(dataset))
    if dataset:
        print("First example:")
        for message in dataset[0].get("messages", []):
            print(message)


def validate_dataset(dataset):
    """
    Valida o formato dos registros do dataset.
    
    Verifica:
     - Se o registro é um dicionário;
     - Se contém a chave 'messages' com uma lista;
     - Se cada mensagem possui as chaves 'role' e 'content';
     - Se 'role' é um dos valores válidos;
     - Se 'content' é uma string;
     - Se há pelo menos uma mensagem do assistente.
    
    :param dataset: Lista de registros.
    :return: Dicionário com contagem de erros por tipo.
    """
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages")
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            # Verifica chaves não reconhecidas
            if any(key not in ("role", "content", "name", "function_call", "weight") for key in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role") not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content")
            function_call = message.get("function_call")
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role") == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    return format_errors


def get_token_encoder(encoding_name="cl100k_base"):
    """
    Retorna a codificação do tiktoken especificada.
    
    :param encoding_name: Nome da codificação.
    :return: Codificação tiktoken.
    """
    return tiktoken.get_encoding(encoding_name)


def num_tokens_from_messages(messages, encoding, tokens_per_message=3, tokens_per_name=1):
    """
    Calcula o número de tokens de uma lista de mensagens.
    
    :param messages: Lista de mensagens (cada uma é um dicionário).
    :param encoding: Codificação do tiktoken.
    :param tokens_per_message: Tokens fixos por mensagem.
    :param tokens_per_name: Tokens adicionais se a chave 'name' estiver presente.
    :return: Número total de tokens.
    """
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # tokens adicionais do sistema
    return num_tokens


def num_assistant_tokens_from_messages(messages, encoding):
    """
    Calcula o número de tokens presentes nas mensagens do assistente.
    
    :param messages: Lista de mensagens.
    :param encoding: Codificação do tiktoken.
    :return: Número de tokens nas mensagens do assistente.
    """
    num_tokens = 0
    for message in messages:
        if message.get("role") == "assistant":
            num_tokens += len(encoding.encode(message.get("content", "")))
    return num_tokens


def print_distribution(values, name):
    """
    Exibe estatísticas de distribuição para uma lista de valores.
    
    :param values: Lista de valores.
    :param name: Nome da distribuição.
    """
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")


def compute_token_statistics(dataset, encoding):
    """
    Calcula estatísticas de tokens para cada exemplo do dataset.
    
    :param dataset: Lista de registros.
    :param encoding: Codificação do tiktoken.
    :return: Tuple contendo:
             - Lista com número de mensagens por exemplo;
             - Lista com total de tokens por exemplo;
             - Lista com tokens das mensagens do assistente;
             - Número de exemplos sem mensagem de sistema;
             - Número de exemplos sem mensagem de usuário.
    """
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex.get("messages", [])
        if not any(message.get("role") == "system" for message in messages):
            n_missing_system += 1
        if not any(message.get("role") == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages, encoding))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages, encoding))

    return n_messages, convo_lens, assistant_message_lens, n_missing_system, n_missing_user


def estimate_epochs(dataset, target_epochs=3, min_target_examples=100,
                    max_target_examples=25000, min_default_epochs=1,
                    max_default_epochs=25):
    """
    Estima o número de épocas para o fine-tuning com base no número de exemplos.
    
    :param dataset: Lista de registros.
    :param target_epochs: Número alvo de épocas.
    :param min_target_examples: Mínimo de exemplos alvo.
    :param max_target_examples: Máximo de exemplos alvo.
    :param min_default_epochs: Mínimo de épocas padrão.
    :param max_default_epochs: Máximo de épocas padrão.
    :return: Número estimado de épocas.
    """
    n_train_examples = len(dataset)
    n_epochs = target_epochs
    if n_train_examples * target_epochs < min_target_examples:
        n_epochs = min(max_default_epochs, min_target_examples // n_train_examples)
    elif n_train_examples * target_epochs > max_target_examples:
        n_epochs = max(min_default_epochs, max_target_examples // n_train_examples)
    return n_epochs


def estimate_billing_tokens(convo_lens, max_tokens_per_example=16385):
    """
    Estima o total de tokens que serão cobrados durante o treinamento.
    
    :param convo_lens: Lista com o total de tokens por exemplo.
    :param max_tokens_per_example: Limite máximo de tokens por exemplo.
    :return: Número total de tokens para cobrança.
    """
    return sum(min(max_tokens_per_example, length) for length in convo_lens)


def estimate_cost(n_epochs, n_billing_tokens, cost_per_thousand=0.025):
    """
    Estima o custo total de treinamento.
    
    :param n_epochs: Número de épocas.
    :param n_billing_tokens: Total de tokens cobrados.
    :param cost_per_thousand: Custo por 1.000 tokens (USD).
    :return: Custo total estimado (USD).
    """
    return n_epochs * n_billing_tokens / 1000 * cost_per_thousand


def main():
    # Caminho do arquivo JSONL (pode ser parametrizado)
    data_path = (
        "/Users/almsantana/Documents/GitHub/4565-Fine-Tuning-OpenAI/"
        "aula_2/data_base_fine_tuning.jsonl"
    )
    dataset = load_dataset(data_path)

    print_initial_stats(dataset)

    format_errors = validate_dataset(dataset)
    if format_errors:
        print("Found errors:")
        for error, count in format_errors.items():
            print(f"{error}: {count}")
    else:
        print("No errors found")

    encoding = get_token_encoder("cl100k_base")
    (n_messages, convo_lens, assistant_message_lens,
     n_missing_system, n_missing_user) = compute_token_statistics(dataset, encoding)

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")

    n_too_long = sum(1 for l in convo_lens if l > 16385)
    print(f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning")

    n_epochs = estimate_epochs(dataset)
    n_billing_tokens = estimate_billing_tokens(convo_lens)
    print(f"Dataset has ~{n_billing_tokens} tokens that will be charged for during training")
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens} tokens")
    cost = estimate_cost(n_epochs, n_billing_tokens, cost_per_thousand=0.025)
    print(f" --> Custo total: {cost:.2f} USD")


if __name__ == "__main__":
    main()
