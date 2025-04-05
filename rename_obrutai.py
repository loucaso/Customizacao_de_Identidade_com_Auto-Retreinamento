import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 🏗️ Caminho do modelo base
MODEL_PATH = "C:/ObrutAi"

# 📌 Carregar dataset de identidade (apenas para mudança de nome/resposta)
dataset = load_dataset("json", data_files="identity_override.jsonl", split="train")

# 🚀 Carregar tokenizador e modelo do caminho local
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, attn_implementation="eager")

# 🔥 Ativar RoPE Scaling para 128K com janela de 8K
model.config.update({
    "rope_scaling": {"type": "linear", "factor": 4},  # 32K * 4 = 128K
    "max_position_embeddings": 131072  # opcional, força contexto máximo
})

# 🛠️ Função de tokenização com max_length 1024 (janela de atenção)
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )
    return {
        "input_ids": result["input_ids"][0],
        "attention_mask": result["attention_mask"][0],
        "labels": result["input_ids"][0]
    }

# 🔍 Tokenizar dataset de identidade
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 🎯 Configuração do primeiro treino (mudar identidade)
training_args = TrainingArguments(
    output_dir="D:/obrutai-identity-override",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    bf16=False
)

# 🔥 Treinamento inicial
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("🚀 Iniciando fine-tuning da identidade...")
trainer.train()

# 🔄 Auto-retreinamento: o modelo aprende com suas próprias respostas
print("🔄 Iniciando auto-retreinamento supervisionado...")

auto_train_dataset = []
for _ in range(100):  # Gera 100 exemplos
    prompt = "Explique um conceito avançado de programação web: "
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=1024)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    auto_train_dataset.append({"text": response})

# 📌 Criar dataset sintético e tokenizar
auto_train_dataset = Dataset.from_list(auto_train_dataset)
tokenized_auto_dataset = auto_train_dataset.map(tokenize, remove_columns=["text"])

# ⚙️ Configuração do auto-retreinamento
auto_training_args = TrainingArguments(
    output_dir="D:/obrutai-self-trained",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=20,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=1e-5,
    weight_decay=0.01,
    fp16=True,
    bf16=False
)

auto_trainer = SFTTrainer(
    model=model,
    args=auto_training_args,
    train_dataset=tokenized_auto_dataset,
)

print("🔥 Treinando modelo com suas próprias previsões...")
auto_trainer.train()

# 💾 Salvar modelo final em D:
model.save_pretrained("D:/ObrutAi-tuned")
tokenizer.save_pretrained("D:/ObrutAi-tuned")

print("✅ Treinamento finalizado! Modelo salvo em D:/ObrutAi-tuned")
