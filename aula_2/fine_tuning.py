from openai import OpenAI
from dotenv import load_dotenv
import os
import time
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")        

client = OpenAI(api_key=openai_api_key)

ft_json_training_dataset = client.files.create(
  file=open("aula_2/data_base_fine_tuning.jsonl", "rb"),
  purpose="fine-tune"
)

ft_json_validating_dataset = client.files.create(
  file=open("aula_2/data_base_validacao.jsonl", "rb"),
  purpose="fine-tune"
)

print("Training ID: ", ft_json_training_dataset.id)
print("Validating ID: ", ft_json_validating_dataset.id)

job = client.fine_tuning.jobs.create(
    training_file=ft_json_training_dataset.id,
    validation_file=ft_json_validating_dataset.id,
    suffix="Aula-Teste-Alura-v0",
    model="gpt-4o-2024-08-06",
    method={
        "type": "supervised",
        "supervised": {
            "hyperparameters": {"n_epochs": 3},
        },
    },
)

state = client.fine_tuning.jobs.retrieve(job.id)

while(state.status is not "completed"):
    print("Job status: ", state.status)
    state = client.fine_tuning.jobs.retrieve(job.id)
    time.sleep(5)



