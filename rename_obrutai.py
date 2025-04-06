import torch
import random
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# 🏗️ Caminho do modelo base
MODEL_PATH = "C:/ObrutAi"

# 📌 Carregar dataset de identidade manual (override inicial)
dataset = load_dataset("json", data_files="identity_override.jsonl", split="train")

# 🚀 Carregar tokenizador e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, attn_implementation="eager")

# 🔍 RoPE Scaling para 64K (janela real de 8K, fator 2)
model.config.update({
    "rope_scaling": {"type": "linear", "factor": 2},
    "max_position_embeddings": 65536
})

# 🧼 Tokenizador (janela de 512 tokens)
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": result["input_ids"][0],
        "attention_mask": result["attention_mask"][0],
        "labels": result["input_ids"][0]
    }

# 🔍 Tokenizar dataset manual de identidade
tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 🎯 Treinamento inicial (apenas identidade)
training_args = TrainingArguments(
    output_dir="C:/obrutai-identity-override",
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

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("🚀 Iniciando fine-tuning da identidade...")
trainer.train()

# 🧠 Prompts mistos: identidade + especialização
prompts = [
    "Quem é você?", "Qual o seu nome?", "Fale como o ObrutAi sobre inteligência artificial.",
    "Simule uma conversa com um programador.", "Explique sua origem e função como modelo da TurboRio.",
    "Responda como um modelo treinado pela DSantos Info.", "Você é o ObrutAi? Prove!",
    "Explique um conceito avançado de programação web.", "Como funciona o protocolo HTTP?",
    "Escreva uma função em PHP que conecta ao MySQL.", "Crie uma explicação sobre WebAssembly.",
    "Responda como se fosse o ObrutAi ensinando sobre web3.",
    "Descreva como construir um jogo HTML5 com backend em PHP.",
    "Explique o Node.js e APIs REST.", "Como usar o MongoDB com JavaScript?",
    "O que são cookies, sessions e tokens?", "Explique como escalar um backend com Node.js e MySQL.",
    "Qual a diferença entre POST e GET?", "Como usar async/await em JavaScript?",
    "Escreva um exemplo de CRUD com MongoDB.",
    "Oi!", "Olá, tudo bem?", "Olá ObrutAi!", "Oi, qual seu nome?", "E aí, quem é você?"
]

# 🔁 Verificação de repetição por n-gram simples
def is_looping(text):
    words = text.split()
    for size in range(4, 20):
        for i in range(len(words) - 2*size):
            pattern = words[i:i+size]
            next_chunk = words[i+size:i+2*size]
            if pattern == next_chunk:
                return True
    return False

# 🔄 Auto-retreinamento: geração supervisionada
print("🔄 Iniciando auto-retreinamento com prompts diversos...")

auto_train_dataset = []
for i in range(100):
    prompt = random.choice(prompts)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Garantir que total não ultrapasse 8192 tokens
    prompt_len = input_ids.shape[-1]
    max_new = min(768, 8192 - prompt_len)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove o prompt da resposta (evita eco)
    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

    # 🪵 Log a cada passo da geração
    print(f"[{i+1}/100] Prompt usado: {prompt}")
    print(f"Resposta gerada:\n{response}\n{'-'*60}")

    # 🧼 Evita respostas em loop
    if not is_looping(response) and len(response) > 20:
        auto_train_dataset.append({"text": f"{prompt}\n{response}"})
    else:
        print("⚠️ Resposta descartada por repetição ou ser muito curta.\n" + "-"*60)

# 📚 Preparar dataset sintético
auto_train_dataset = Dataset.from_list(auto_train_dataset)
tokenized_auto_dataset = auto_train_dataset.map(tokenize, remove_columns=["text"])

# 🧪 Argumentos de auto-retreinamento
auto_training_args = TrainingArguments(
    output_dir="C:/obrutai-self-trained",
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

print("🔥 Treinando modelo com suas próprias previsões e especialização web...")
auto_trainer.train()

# 💾 Salvar modelo final
model.save_pretrained("C:/ObrutAi-tuned")
tokenizer.save_pretrained("C:/ObrutAi-tuned")

print("✅ Treinamento finalizado! Modelo salvo em C:/ObrutAi-tuned")
