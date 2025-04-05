
# ObrutAi — Customização de Identidade com Auto-Retreinamento

Este projeto realiza o **ajuste fino da identidade** de um modelo baseado no modelos llm com RoPE Scaling para até 128K tokens, usando **auto-retreinamento supervisionado** a partir das próprias respostas do modelo.

---

## ✅ Objetivo

- Alterar a forma como o modelo responde à pergunta “Quem é você?”
- Manter coerência em janelas de 8.192 tokens com RoPE scaling para 128K
- Treinar o modelo usando os próprios outputs, criando uma base de conhecimento personalizada

---

## 📁 Estrutura esperada

```
ObrutAi/
│
├── rename_obrutai.py                # Script principal de treinamento
├── identity_override.jsonl         # Dataset com override de identidade
└── [modelo base ]        # Pasta com modelo base já baixado
```

---

## 📄 Como funciona o dataset `identity_override.jsonl`

Este arquivo contém um JSON por linha com prompts personalizados. Exemplo:

```json
{"text": "Quem é você?"}
{"text": "Qual sua origem?"}
{"text": "Você foi treinado por quem?"}
```

O objetivo é que o modelo aprenda a responder essas perguntas com a **identidade definida por você**, como:

> "Fui retreinado pela equipe da xxxx a partir do xxx."

---

## 💻 Instalação das dependências

Recomenda-se o uso de um ambiente virtual com Python 3.10+.

```bash
pip install -U torch transformers accelerate datasets trl
```

Para Windows com CUDA (GPU NVIDIA), certifique-se que o PyTorch com suporte a CUDA esteja instalado:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Como rodar o código

1. Coloque seu modelo base dentro da pasta `C:/ObrutAi`
2. Crie o arquivo `identity_override.jsonl` com as perguntas que deseja alterar a resposta
3. Execute o script principal:

```bash
python rename_obrutai.py
```

O modelo final será salvo em:

```
D:/ObrutAi-tuned
```

---

## ⚙️ O que pode ser alterado

- `identity_override.jsonl`: você pode trocar ou adicionar prompts conforme desejar
- Número de exemplos gerados no auto-retreinamento (`range(100)`)
- Quantidade de tokens gerados por resposta (`max_new_tokens`)
- Número de épocas (`num_train_epochs`)
- Caminhos de entrada e saída
- Parâmetros de RoPE (ex: fator linear)

---

## 🔍 Explicação das funções

### 1. `tokenize(example)`
Tokeniza cada entrada de texto, ajustando para `max_length=8192` (janela de atenção). Usa padding fixo e retorna também as labels para treinamento supervisionado.

---

### 2. `RoPE Scaling`
```python
model.config.update({
  "rope_scaling": {"type": "linear", "factor": 4}
})
```
Permite ao modelo extrapolar seu limite de contexto para até 128K tokens, mesmo mantendo a janela de atenção em 8K.

---

### 3. `trainer = SFTTrainer(...)`
Executa o primeiro fine-tuning, apenas para alterar as respostas de identidade.

---

### 4. `auto_train_dataset.append(...)`
Gera novas amostras de texto com prompts e as respostas do próprio modelo. Isso é usado no **auto-retreinamento**.

---

### 5. `auto_trainer.train()`
Treina o modelo usando os exemplos sintéticos criados a partir das respostas geradas.

---

### 6. `model.save_pretrained(...)`
Salva o modelo e o tokenizador finalizado em `D:/ObrutAi-tuned`.

---

## 📞 Suporte

Se você tiver dúvidas, sugestões ou quiser integrar novos tipos de memória (como memória vetorial, pinecone ou RAG), fale com a equipe TurboRio.

---
