# 🛑 Early Terminate Hyperband - Explicação Detalhada

## 🎯 O que é Early Terminate?

**Early Terminate** é uma técnica que **para automaticamente** runs do sweep que não estão performando bem, economizando tempo e recursos computacionais.

## 🔧 Como funciona o Hyperband?

O **Hyperband** é um algoritmo que:

1. **Inicia muitos runs** com poucos recursos (épocas)
2. **Avalia performance** de cada run
3. **Mata os piores** e continua apenas com os melhores
4. **Dá mais recursos** (mais épocas) aos sobreviventes
5. **Repete o processo** até encontrar os melhores hiperparâmetros

## 📊 Configuração atual no sweep_config.yaml:

```yaml
early_terminate:
  type: hyperband
  min_iter: 4     # Mínimo 4 épocas antes de terminar
  max_iter: 20    # Máximo 20 épocas
  s: 2            # Fator de redução agressiva
  eta: 3          # Fator de eliminação (mata 2/3 dos runs)

# IMPORTANTE: Command SEM --epochs fixo para hyperband funcionar
command: ["python", "main.py", "${args}"]
```

## 🎮 Exemplo Prático de Como Funciona:

### **Rodada 1** (Todas começam com 4 épocas):
```
Run 1: f1_precision_combined = 0.65  ✅ Continua
Run 2: f1_precision_combined = 0.45  ❌ ELIMINADO
Run 3: f1_precision_combined = 0.70  ✅ Continua  
Run 4: f1_precision_combined = 0.40  ❌ ELIMINADO
Run 5: f1_precision_combined = 0.68  ✅ Continua
Run 6: f1_precision_combined = 0.35  ❌ ELIMINADO
```

### **Rodada 2** (Sobreviventes ganham mais épocas: 4 → 12):
```
Run 1: f1_precision_combined = 0.72  ✅ Continua
Run 3: f1_precision_combined = 0.74  ✅ Continua
Run 5: f1_precision_combined = 0.69  ❌ ELIMINADO
```

### **Rodada 3** (Finalistas ganham épocas máximas: 12 → 20):
```
Run 1: f1_precision_combined = 0.78  
Run 3: f1_precision_combined = 0.81  ← VENCEDOR!
```

## ⚙️ Parâmetros Explicados:

### **`min_iter: 4`**
- **O que faz**: Todo run deve treinar pelo menos 4 épocas
- **Por que**: Evita eliminar runs que podem precisar de "aquecimento"
- **Impacto**: Quanto menor, mais agressivo (mas pode eliminar bons runs cedo)

### **`max_iter: 20`**
- **O que faz**: Nenhum run pode passar de 20 épocas
- **Por que**: Evita desperdício de tempo em runs que não convergem
- **Impacto**: Limite máximo de tempo por run

### **`s: 2`**
- **O que faz**: Controla quantas "rodadas" de eliminação haverá
- **Por que**: Balanceia exploração vs exploração
- **Valores típicos**: 1-4 (maior = mais rodadas, eliminação mais gradual)

### **`eta: 3`**
- **O que faz**: Em cada rodada, elimina 2/3 dos runs (mantém 1/3)
- **Por que**: Eliminação agressiva mas não muito radical
- **Exemplo**: 
  - 18 runs → 6 runs → 2 runs → 1 vencedor
  - eta=2: mata metade (menos agressivo)
  - eta=4: mata 3/4 (mais agressivo)

## 🏆 Vantagens do Early Terminate:

### ✅ **Economia de Tempo**
```
❌ Sem early terminate: 50 runs × 20 épocas = 1000 épocas totais
✅ Com early terminate: ~150-200 épocas totais (5x mais rápido!)
```

### ✅ **Foco nos Melhores**
- Elimina rapidamente hiperparâmetros ruins
- Dedica mais tempo aos promissores
- Encontra o ótimo mais eficientemente

### ✅ **Previne Overfitting**
- Runs ruins são cortados antes de overfittarem
- Foco em convergência saudável

## 🎯 Configurações Recomendadas por Cenário:

### **🚀 Sweep Rápido (teste)**
```yaml
early_terminate:
  type: hyperband
  min_iter: 2     # Muito agressivo
  max_iter: 10    # Limite baixo
  s: 1            # Poucas rodadas
  eta: 4          # Eliminação muito agressiva
```
**Resultado**: Muito rápido, pode perder bons hiperparâmetros

### **⚖️ Sweep Balanceado (atual)**
```yaml
early_terminate:
  type: hyperband
  min_iter: 3     # Moderadamente agressivo
  max_iter: 50    # Limite médio
  s: 2            # Rodadas balanceadas
  eta: 3          # Eliminação moderada
```
**Resultado**: Bom balanço tempo vs qualidade

### **🎯 Sweep Conservativo (qualidade máxima)**
```yaml
early_terminate:
  type: hyperband
  min_iter: 5     # Menos agressivo
  max_iter: 100   # Limite alto
  s: 3            # Mais rodadas
  eta: 2          # Eliminação menos agressiva
```
**Resultado**: Mais lento, mas melhor chance de encontrar o ótimo global

## 📊 Monitoramento no W&B:

Durante o sweep você verá:

```
🔴 Run abc123: Stopped early at epoch 3 (performance: 0.45)
🟡 Run def456: Continuing to epoch 9 (performance: 0.68) 
🟢 Run ghi789: Advanced to epoch 27 (performance: 0.75)
```

## 🛠️ Personalizando para seu Caso:

### **Para otimizar PRECISION** (foco atual):
```yaml
early_terminate:
  type: hyperband
  min_iter: 4     # Dar tempo para pesos se ajustarem
  max_iter: 30    # Suficiente para convergência de precision
  s: 2            # Balanceado
  eta: 3          # Padrão
```

### **Para modelos grandes (ESM2)**:
```yaml
early_terminate:
  type: hyperband
  min_iter: 2     # Modelos grandes convergem rápido
  max_iter: 20    # Evitar overfitting
  s: 2
  eta: 4          # Mais agressivo (memória limitada)
```

## 🚨 Quando NÃO usar Early Terminate:

1. **Runs muito curtos** (< 5 épocas): Não há tempo para julgar
2. **Métricas instáveis**: Se a métrica varia muito no início
3. **Debugging**: Quando quer ver o comportamento completo

Para desabilitar:
```yaml
# Simplesmente remover a seção early_terminate do YAML
```

## 💡 Dicas Práticas:

1. **Monitore os logs**: Veja quais runs estão sendo eliminados
2. **Ajuste baseado em resultados**: Se muitos bons runs morrem cedo, aumente `min_iter`
3. **Use métricas estáveis**: `f1_precision_combined` é boa para early terminate
4. **Comece conservativo**: Primeiro sweep com early terminate suave

## 🚨 IMPORTANTE: Por que NÃO pode ter --epochs fixo?

### ❌ **Configuração ERRADA** (Early terminate não funciona):
```yaml
command: ["python", "main.py", "--epochs", "5", "${args}"]
early_terminate: {...}
```
**Problema**: W&B não consegue override o `--epochs` especificado no command!

### ✅ **Configuração CORRETA** (Early terminate funciona):
```yaml  
command: ["python", "main.py", "${args}"]  # SEM --epochs
early_terminate: {...}
```
**Resultado**: W&B Hyperband controla épocas dinamicamente via `config_wandb._epochs`

### 🎯 **Como funciona internamente**:
1. **W&B define épocas**: `config_wandb._epochs = 4` (primeira rodada)
2. **Trainer usa essas épocas**: `trainer.epochs = 4`  
3. **Avalia performance**: `f1_precision_combined = 0.65`
4. **W&B decide**: "Continuar!" → `config_wandb._epochs = 12`
5. **Repete até max_iter**: ou até ser eliminado

---

**🎯 Resumo**: O early terminate é como um "torneio de eliminação" que encontra os melhores hiperparâmetros de forma eficiente, evitando desperdiçar tempo com configurações ruins! 