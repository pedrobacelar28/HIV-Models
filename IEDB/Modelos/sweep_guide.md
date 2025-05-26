# 🎯 Guia do W&B Hyperparameter Sweep

Este guia explica como usar o **Weights & Biases Hyperparameter Sweep** para otimizar automaticamente os pesos da loss function e melhorar a **precision** do modelo.

## 🚀 Como usar o Sweep

### 1. Configurar defaults no main.py

Edite os defaults no `main.py` para definir o dataset e modelo:

```python
# ────── Configuração básica ──────────────────────────────────────
DEFAULT_DATASET = "B"                      # Dataset para otimizar
DEFAULT_MODEL = "esmc_300m"                 # Modelo a usar
DEFAULT_VIRUS_TYPE = "Base"                 # Tipo de arquivo
DEFAULT_EPOCHS = 5                         # Épocas por run (rápido para sweep)
```

### 2. Ativar o Sweep

```bash
# Método 1: Editar default
# DEFAULT_SWEEP_MODE = True
python main.py

# Método 2: Argumento
python main.py --sweep
```

### 3. Acompanhar o progresso

O sweep criará automaticamente um projeto W&B e exibirá o link:

```
✅ Sweep criado com ID: abc123
🌐 Acompanhe em: https://wandb.ai/sua-conta/protein-b-sweep/sweeps/abc123
```

## ⚙️ O que o Sweep otimiza

### 🎯 Métrica Principal
- **`test/f1_precision_combined`**: Média harmônica entre F1 e Precision
- **Objetivo**: Maximizar (foco em precision sem perder F1)

### 🔧 Hiperparâmetros otimizados

**Pesos da Loss Function** (foco principal):
- `pos_class_weight`: 1.0 → 10.0 (peso para classe negativa)
- `loss_weight_multiplier`: 0.5 → 3.0 (multiplicador geral)

**Treinamento** (secundário):
- `lr`: 1e-6 → 1e-3 (learning rate)
- `weight_decay`: 0.0 → 0.1 (regularização L2)
- `dropout`: 0.0 → 0.5 (regularização do modelo)

### 🚫 O que NÃO é otimizado

- `batch_size`: Mantido fixo para estabilidade
- `max_length`: Mantido fixo para comparabilidade
- `epochs`: Fixo em 5 para sweep rápido

## 📊 Configuração do Sweep (sweep_config.yaml)

```yaml
# Método de otimização
method: bayes              # Busca bayesiana (inteligente)

# Métrica a maximizar
metric:
  name: test/f1_precision_combined
  goal: maximize

# Número máximo de runs
run_cap: 50

# Early termination (evita runs ruins)
early_terminate:
  type: hyperband
  min_iter: 3             # Mínimo 3 épocas
```

## 🎯 Estratégia de Otimização

### Para melhorar **Precision** (reduzir falsos positivos):

1. **`pos_class_weight` > 1.0**: Penaliza mais erros na classe negativa
2. **`loss_weight_multiplier` > 1.0**: Amplifica o efeito dos pesos
3. **`weight_decay` > 0.0**: Regularização para evitar overfitting

### Resultado esperado:
- ✅ Menos predições positivas incorretas
- ✅ Maior confiabilidade nas predições positivas
- ⚠️ Possível redução leve no recall

## 📈 Interpretando os Resultados

### Métricas no W&B Dashboard:

**Principais**:
- `test/f1_precision_combined` ⬆️ (objetivo principal)
- `test/precision` ⬆️ (foco específico)
- `test/f1` ➡️ (manter alto)

**Secundárias**:
- `test/recall` ⬇️ (pode diminuir levemente)
- `test/accuracy` ➡️ (estabilidade geral)

### Matriz de Confusão ideal:
```
              Predito
           Neg   Pos
Real  Neg  ↑TN   ↓FP   ← Foco: reduzir FP
      Pos  →FN   →TP
```

## 🔧 Personalizando o Sweep

### Modificar faixa de hiperparâmetros:

Edite `sweep_config.yaml`:

```yaml
parameters:
  pos_class_weight:
    min: 2.0      # Começar já com peso alto
    max: 15.0     # Testar pesos muito altos
  
  loss_weight_multiplier:
    min: 1.0      # Sem redução de efeito
    max: 5.0      # Amplificação muito alta
```

### Adicionar novos hiperparâmetros:

```yaml
parameters:
  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5
```

## 🚀 Exemplos de Uso

### 1. Sweep completo padrão:
```bash
python main.py --sweep --dataset B --model esmc_300m
```

### 2. Sweep com configuração específica:
```bash
python main.py --sweep \
    --dataset MHC1 \
    --model esmc_600m \
    --virus-type Base \
    --epochs 3 \
    --wandb-project "meu-sweep-precision"
```

### 3. Sweep rápido para testes:
```bash
# Edite sweep_config.yaml:
# run_cap: 10
python main.py --sweep
```

## 📋 Checklist para usar o Sweep

- [ ] Configurar defaults no `main.py`
- [ ] Verificar que `sweep_config.yaml` existe
- [ ] Ter conta W&B configurada (`wandb login`)
- [ ] Executar `python main.py --sweep`
- [ ] Acompanhar progresso no dashboard W&B
- [ ] Analisar resultados e aplicar melhores hiperparâmetros

## 🏆 Melhores Práticas

1. **Primeiro**: Execute sweep curto (10 runs) para testar
2. **Depois**: Execute sweep completo (50 runs) 
3. **Analise**: Use o dashboard W&B para identificar padrões
4. **Aplique**: Use os melhores hiperparâmetros em treino final
5. **Valide**: Teste o modelo otimizado no conjunto de teste

## 🐛 Resolução de Problemas

### "Sweep config not found":
```bash
# Verificar se arquivo existe
ls sweep_config.yaml

# Recriar se necessário
curl -O https://raw.githubusercontent.com/.../sweep_config.yaml
```

### "W&B not logged in":
```bash
wandb login
# Cole sua API key quando solicitado
```

### "Out of memory":
```bash
# Reduzir batch size no main.py
DEFAULT_BATCH_SIZE = 4  # ou menor
```

---

💡 **Dica**: O sweep pode levar algumas horas dependendo do número de runs. Use `run_cap: 10` para testes rápidos! 