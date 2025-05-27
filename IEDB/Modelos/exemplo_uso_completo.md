# 🚀 Guia Completo: Pré-treinamento + Fine-tuning ESM-C

## 📋 Respostas às Suas Perguntas

### 1. 📁 **Diferença entre Checkpoint e Backbone**

```bash
# Estrutura após pré-treinamento:
MHC2/pretraining/pretrain_esmc_600m_20241218_143022/
├── checkpoint_epoch_2.pt           # ✅ Para CONTINUAR pré-treinamento
├── checkpoint_epoch_4.pt           # ✅ Para CONTINUAR pré-treinamento  
├── checkpoint_epoch_6.pt           # ✅ Para CONTINUAR pré-treinamento
├── checkpoint_epoch_10.pt          # ✅ Para CONTINUAR pré-treinamento
├── pretrained_esmc_epoch_10.pt     # ✅ Para FINE-TUNING
└── best_pretrained_esmc.pt         # ✅ Para FINE-TUNING (melhor)
```

**Checkpoint** = Estado completo (modelo + otimizador + scheduler)  
**Backbone** = Apenas pesos do ESM-C (para classificação)

### 2. 🎯 **Fine-tuning Após Pré-treinamento**

```python
# Configure no main.py:
DEFAULT_PRETRAIN_MODE = False  # ❌ Não pré-treinar
DEFAULT_PRETRAINED_BACKBONE_PATH = "./MHC2/pretraining/pretrain_esmc_600m_20241218_143022/best_pretrained_esmc.pt"  # ✅
```

**Ou via linha de comando:**
```bash
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --pretrained-backbone-path ./MHC2/pretraining/pretrain_esmc_600m_20241218_143022/best_pretrained_esmc.pt \
    --epochs 15
```

### 3. 🔄 **Continuar Pré-treinamento por Mais Épocas**

```bash
# Continuar de onde parou:
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --resume-pretraining ./MHC2/pretraining/pretrain_esmc_600m_20241218_143022/checkpoint_epoch_10.pt \
    --pretrain-epochs 20  # Total de épocas desejadas
```

---

## 🔬 Cenários Práticos

### **Cenário A: Primeiro Pré-treinamento**

```bash
# 1. Pré-treinar por 10 épocas
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --pretrain \
    --pretrain-epochs 10 \
    --pretrain-lr 5e-5 \
    --pretrain-batch-size 128

# Resultado: Salva em ./MHC2/pretraining/pretrain_esmc_600m_TIMESTAMP/
```

### **Cenário B: Continuar Pré-treinamento**

```bash
# 2. Listar checkpoints disponíveis
python main.py --dataset MHC2 --model esmc_600m --list-pretraining

# 3. Continuar por mais 10 épocas (total 20)
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --resume-pretraining ./MHC2/pretraining/pretrain_esmc_600m_TIMESTAMP/checkpoint_epoch_10.pt \
    --pretrain-epochs 20
```

### **Cenário C: Fine-tuning com Backbone**

```bash
# 4. Fine-tuning usando backbone pré-treinado
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --pretrained-backbone-path ./MHC2/pretraining/pretrain_esmc_600m_TIMESTAMP/best_pretrained_esmc.pt \
    --epochs 15 \
    --lr 1e-5
```

---

## 🎯 Workflows Recomendados

### **Workflow 1: Pré-treinamento Incremental**

```bash
# Etapa 1: Pré-treinar 5 épocas (teste rápido)
python main.py --dataset MHC2 --model esmc_600m --pretrain --pretrain-epochs 5

# Verificar W&B - se loss diminuindo bem:
# Etapa 2: Continuar por mais 10 épocas (total 15)
python main.py --dataset MHC2 --model esmc_600m \
    --resume-pretraining ./MHC2/pretraining/pretrain_esmc_600m_*/checkpoint_epoch_5.pt \
    --pretrain-epochs 15

# Etapa 3: Fine-tuning
python main.py --dataset MHC2 --model esmc_600m \
    --pretrained-backbone-path ./MHC2/pretraining/pretrain_esmc_600m_*/best_pretrained_esmc.pt \
    --epochs 10
```

### **Workflow 2: Automático (Recomendado)**

```bash
# Tudo em uma execução: 10 épocas pré-treino + 15 fine-tuning
python main.py \
    --dataset MHC2 \
    --model esmc_600m \
    --pretrain \
    --pretrain-epochs 10 \
    --epochs 15
```

---

## 🔍 Comandos Úteis

### **Listar Modelos Disponíveis**

```bash
# Listar runs de fine-tuning
python main.py --dataset MHC2 --model esmc_600m --list-runs

# Listar checkpoints de pré-treinamento
python main.py --dataset MHC2 --model esmc_600m --list-pretraining
```

### **Monitoramento**

```bash
# Monitorar pré-treinamento
# W&B projeto: esmc-pretrain-mhc2

# Monitorar fine-tuning  
# W&B projeto: protein-mhc2
```

---

## ⚡ Configurações Otimizadas

### **Para Dataset MHC2**

```bash
# Pré-treinamento otimizado
--pretrain-max-length 60 \
--pretrain-batch-size 128 \
--pretrain-lr 5e-5 \
--pretrain-epochs 10

# Fine-tuning otimizado
--batch-size 128 \
--lr 1e-5 \
--epochs 15
```

### **Para Dataset com Poucos Dados**

```bash
# Pré-treinamento mais conservador
--pretrain-epochs 5 \
--pretrain-lr 1e-5 \
--pretrain-batch-size 64

# Fine-tuning com early stopping
--epochs 20  # W&B vai parar se não melhorar
```

---

## 🐛 Soluções para Problemas Comuns

### **1. "Out of memory" durante pré-treinamento**

```bash
# Reduzir batch size e max length
--pretrain-batch-size 32 \
--pretrain-max-length 40
```

### **2. Loss MLM não diminui**

```bash
# Reduzir learning rate
--pretrain-lr 1e-5  # Em vez de 5e-5
```

### **3. Checkpoint não encontrado**

```bash
# Verificar caminhos disponíveis
python main.py --dataset MHC2 --model esmc_600m --list-pretraining

# Usar caminho completo
--resume-pretraining /caminho/completo/para/checkpoint_epoch_X.pt
```

### **4. Comparar performance**

```bash
# Baseline (sem pré-treinamento)
python main.py --dataset MHC2 --model esmc_600m --epochs 15 --wandb-project comparison

# Com pré-treinamento
python main.py --dataset MHC2 --model esmc_600m --pretrain --pretrain-epochs 10 --epochs 15 --wandb-project comparison
```

---

## 🏆 Expectativas de Melhoria

### **Sinais de Sucesso**

- [ ] MLM loss diminui de ~6.0 para ~2.0-3.0
- [ ] Fine-tuning converge em menos épocas
- [ ] F1 score 2-5% maior que baseline
- [ ] Precision melhor (menos falsos positivos)
- [ ] Treinamento mais estável

### **Quando Usar Pré-treinamento**

✅ **Use quando:**
- Dataset de epitopos < 10k sequências
- Epitopos muito curtos (8-12 aminoácidos)
- Quero melhor precision
- Tenho tempo para experimentar

❌ **Não use quando:**
- Dataset muito grande (>100k)
- Epitopos longos (>20 aminoácidos)
- Deadline apertado
- Baseline já está excelente

---

## 📊 Monitoramento Avançado

### **Métricas de Pré-treinamento (W&B)**

```python
# train/loss - deve diminuir consistentemente
# train/learning_rate - schedule cosine
# train/step - progresso total
```

### **Comparação Final**

| Métrica | Baseline | Com Pré-treino | Melhoria |
|---------|----------|----------------|----------|
| F1 Score | 0.75 | 0.78 | +3% |
| Precision | 0.70 | 0.76 | +6% |
| Convergência | 12 épocas | 8 épocas | -33% |

---

**🎯 Resumo: Suas perguntas foram respondidas e agora você tem um sistema completo para experimentar com pré-treinamento incremental!** 