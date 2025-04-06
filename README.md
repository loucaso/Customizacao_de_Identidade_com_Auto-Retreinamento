
# 🚀 ObrutAi - Fine-Tuning com Identidade e Especialização em Web

Este projeto ajusta um modelo de linguagem baseado no **Gemma 3 1B IT** (Hugging Face) para criar o **ObrutAi**, com foco em:

- 🔗 Identidade fixa e coerente
- 🌐 Especialização em programação web (HTML5, PHP, Web3, MySQL, Node.js, MongoDB)
- 🧠 Contexto expandido para **64K tokens com janela de atenção de 8K**
- ♻️ Auto-retreinamento com geração supervisionada

---

## 🖥️ Ambiente de Treinamento

**Servidor utilizado:**

| Recurso             | Valor                          |
|---------------------|---------------------------------|
| CPU                 | 2x Intel Xeon E5-2680 v3 (24 threads) |
| RAM                 | 80 GB DDR4                     |
| GPU                 | NVIDIA RTX 3060 (6 GB VRAM)    |
| Armazenamento       | 500GB NVMe SSD + 13TB HDDs     |
| Sistema Operacional | Windows 10                     |
| Conexão             | Link dedicado de 1 Gbps        |

**Tempo de treinamento registrado:**

- ⏱️ **~4 horas para 32/100 steps**
- Dataset de identidade simples + 100 prompts para auto-retreinamento

---

## 🔧 Instalação de Dependências

**Pré-requisitos:**
- Python 3.10 ou superior
- NVIDIA GPU com drivers e CUDA instalados

### 1. Crie o ambiente virtual:
```bash
python -m venv obrutai-env
obrutai-env\Scripts\activate
```

### 2. Instale as dependências:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate trl
```

### 3. (Opcional) Para GPUs com baixa VRAM:
```bash
pip install bitsandbytes
```

---

## 📘 Explicação dos Parâmetros (Script `fine_tune_obrutai.py`)

### 🔹 Etapa 1: Treinamento de Identidade
```python
TrainingArguments(
    num_train_epochs=1,             # Aumente para reforçar mais a identidade
    per_device_train_batch_size=2,  # Aumente se tiver mais VRAM (4GB+ por batch)
    learning_rate=2e-5,             # Diminuir = mais estável, aumentar = mais agressivo
    fp16=True                       # Usa menos VRAM (bom para placas de 6GB)
)
```

### 🔹 RoPE Scaling (Expansão de contexto)
```python
model.config.update({
    "rope_scaling": {"type": "linear", "factor": 2},
    "max_position_embeddings": 65536
})
```
- Amplia o modelo para **64K tokens de contexto** mantendo atenção local de 8K.

### 🔹 Geração supervisionada (auto-retreinamento)
```python
max_new = min(768, 8192 - prompt_len)
temperature=0.7
top_p=0.9
repetition_penalty=1.2
```
- Aumente `max_new` para gerar respostas mais longas (↔️ mais lento)
- Reduza `temperature` para respostas mais exatas
- Reduza `top_p` para menos criatividade

---

## ⚙️ Como Rodar o Treinamento

1. Crie o arquivo `identity_override.jsonl` com textos de identidade do modelo.
   - Exemplo:
     ```json
     {"text": "Fui retreinado pela equipe da TurboRio a partir do Gemma 3."}
     ```

2. Salve o modelo base Gemma 3 1B IT no caminho:
   ```
   C:/ObrutAi
   ```

3. Execute o script:
   ```bash
   python fine_tune_obrutai.py
   ```

4. O modelo final será salvo em:
   ```
   C:/ObrutAi-tuned
   ```

---

## ✅ Resultados

- Modelo com **identidade fixada**, responde corretamente à pergunta:
  > "Quem é você?" → *"Fui retreinado pela equipe da TurboRio a partir do Gemma 3."*
- Aprendeu temas como:
  - PHP + MySQL
  - MongoDB com JavaScript
  - Web3 e WebAssembly
  - HTML5 com backend

---

## 📈 Melhoria Contínua

Para acelerar ou melhorar o desempenho:
- Ajuste `batch_size`, `epochs` e `learning_rate`
- Use quantização (`bitsandbytes`)
- Experimente checkpoints intermediários com `save_steps`

---

## 📩 Contato

Projeto da equipe **TurboRio**  
Ajustado por [TurboRio / DSantos Info]
