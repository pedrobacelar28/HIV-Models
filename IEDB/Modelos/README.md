# Sistema Generalizado de Treinamento de Modelos para HIV

Este sistema permite treinar modelos de proteínas de forma flexível, suportando diferentes datasets e tipos de modelos através de parâmetros **TOTALMENTE CONFIGURÁVEIS** no `main.py`.

## 🚀 Características

- **Datasets suportados**: B, MHC1, MHC2
- **Modelos suportados**: 
  - ESMC: `esmc_300m`, `esmc_600m`
  - ESM2: `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`
- **Tipos de vírus**: Tudo, Virus, Lent, Retro
- **🆕 HIPERPARÂMETROS TOTALMENTE CONFIGURÁVEIS** no `main.py`
- **Controle completo** de learning rate, batch size, dropout, etc.
- **Conjuntos de teste específicos HIV** para avaliação independente
- **Critério inteligente** para melhor modelo (F1 + Precision combinados)
- **Logging com Weights & Biases**
- **Suporte a avaliação e treinamento**

## 🆕 NOVA FUNCIONALIDADE: Hiperparâmetros Configuráveis

**AGORA VOCÊ TEM CONTROLE TOTAL** sobre todos os hiperparâmetros diretamente no `main.py`!

### 🔧 Duas formas de configurar:

1. **Editando defaults no `main.py`** (Recomendado para uso frequente)
2. **Argumentos da linha de comando** (Para overrides específicos)

### 📝 Configuração de Defaults no main.py

Edite a seção no início do `main.py` para seus valores preferidos:

```python
# ────── Configuração básica ──────────────────────────────────────
DEFAULT_DATASET = "B"                          # Dataset preferido
DEFAULT_MODEL = "esmc_300m"                     # Modelo preferido
DEFAULT_VIRUS_TYPE = "Base"                     # Tipo de arquivo
DEFAULT_EVAL_MODE = False                       # Modo padrão

# ────── Hiperparâmetros de treinamento ───────────────────────────
DEFAULT_EPOCHS = 30                            # Número de épocas
DEFAULT_LEARNING_RATE = 1e-4                   # Taxa de aprendizado
DEFAULT_WEIGHT_DECAY = 0.01                    # Regularização L2
DEFAULT_BATCH_SIZE = None                      # None = auto
DEFAULT_MAX_LENGTH = None                      # None = auto

# ────── Configuração do modelo ───────────────────────────────────
DEFAULT_DROPOUT = None                         # None = auto
DEFAULT_FREEZE_BACKBONE = False                # Congelar encoder

# ────── Pesos da loss function para precision ───────────────────
DEFAULT_POS_CLASS_WEIGHT = 3.0                # > 1.0 = melhor precision
DEFAULT_LOSS_WEIGHT_MULTIPLIER = 1.0          # Multiplicador adicional
```

**Depois execute simplesmente:** `python main.py`

## 🆕 W&B Hyperparameter Sweep para Otimização Automática

**NOVO**: Sistema de otimização automática de hiperparâmetros focado em **melhorar precision**!

### 🎯 O que é o Sweep?

O **W&B Hyperparameter Sweep** testa automaticamente diferentes combinações de hiperparâmetros para encontrar a configuração que **maximiza a precision** sem perder F1-score.

### 🚀 Como usar:

#### Método 1: Editar defaults
```python
# No main.py:
DEFAULT_SWEEP_MODE = True
DEFAULT_DATASET = "B"
DEFAULT_MODEL = "esmc_300m"
DEFAULT_EPOCHS = 5  # Rápido para sweep
```
```bash
python main.py  # Inicia sweep automaticamente
```

#### Método 2: Argumento direto
```bash
python main.py --sweep --dataset B --model esmc_300m
```

### 🔧 O que o Sweep otimiza:

**Foco Principal** (pesos da loss function):
- `pos_class_weight`: 1.0 → 10.0 (peso para melhorar precision)
- `loss_weight_multiplier`: 0.5 → 3.0 (multiplicador de efeito)

**Secundário**:
- `learning_rate`: 1e-6 → 1e-3 (taxa de aprendizado)
- `weight_decay`: 0.0 → 0.1 (regularização L2)
- `dropout`: 0.0 → 0.5 (regularização do modelo)

**Mantido fixo** (para estabilidade):
- `batch_size`, `max_length`, `epochs`

### 📊 Métrica otimizada:
- **`f1_precision_combined`**: Média harmônica de F1 e Precision
- **Objetivo**: Precision alta SEM perder F1

### 🌐 Acompanhamento:
O sweep cria automaticamente um projeto W&B e exibe o link para acompanhar:
```
✅ Sweep criado com ID: abc123
🌐 Acompanhe em: https://wandb.ai/sua-conta/protein-b-sweep/sweeps/abc123
```

### 📋 Arquivos do Sweep:
- `sweep_config.yaml` - Configuração dos hiperparâmetros
- `sweep_guide.md` - Guia detalhado de uso

**Depois execute simplesmente:** `python main.py`

## 📁 Estrutura de Arquivos

```
Modelos/
├── main.py              # Script principal
├── config.py            # Configurações generalizadas
├── trainer.py           # Classe de treinamento
├── models.py            # Definições de modelos
├── dataset.py           # Dataset personalizado
├── B/                   # Dataset B
│   ├── simB*.txt       # Arquivos positivos de treino
│   ├── naoB*.txt       # Arquivos negativos de treino
│   └── model/          # Modelos salvos
├── MHC1/               # Dataset MHC1
│   ├── simMHC1*.txt    # Arquivos positivos de treino
│   ├── naoMHC1*.txt    # Arquivos negativos de treino
│   └── model/          # Modelos salvos
├── MHC2/               # Dataset MHC2
│   ├── simMHC2*.txt    # Arquivos positivos de treino
│   ├── naoMHC2*.txt    # Arquivos negativos de treino
│   └── model/          # Modelos salvos
└── ../Inferencia_HIV/  # Conjuntos de teste específicos
    ├── Bcell/
    │   ├── simBHIV.txt     # Teste positivo B
    │   └── naoBHIV.txt     # Teste negativo B
    ├── MHC1/
    │   ├── simMHC1HIV.txt  # Teste positivo MHC1
    │   └── naoMHC1HIV.txt  # Teste negativo MHC1
    └── MHC2/
        ├── simMHC2HIV.txt  # Teste positivo MHC2
        └── naoMHC2HIV.txt  # Teste negativo MHC2
```

## 🛠️ Como Usar

### 🆕 Configuração Flexível de Hiperparâmetros

**NOVO**: Agora você controla TODOS os hiperparâmetros diretamente no `main.py`!

#### Método 1: Editar Defaults (Recomendado)

Edite os defaults no topo do `main.py` e execute simplesmente:

```bash
python main.py  # Usa seus defaults configurados
```

#### Método 2: Argumentos da Linha de Comando

Override qualquer hiperparâmetro específico:

```bash
# Controle completo via argumentos
python main.py \
    --dataset B \
    --model esmc_300m \
    --virus-type Base \
    --epochs 20 \
    --lr 5e-4 \
    --batch-size 6 \
    --max-length 80 \
    --dropout 0.3 \
    --pos-class-weight 2.5
```

### Configuração Rápida (sem argumentos)

Para facilitar o uso, edite a seção de defaults no topo do `main.py`:

```python
# 🔧 CONFIGURAÇÃO DE DEFAULTS
DEFAULT_DATASET = "B"                    # "B", "MHC1", "MHC2"
DEFAULT_MODEL = "esmc_300m"              # Modelo a usar
DEFAULT_VIRUS_TYPE = "Base"              # "Base", "Tudo", "Virus", "Lent", "Retro"
DEFAULT_EPOCHS = 30                      # Número de épocas
DEFAULT_EVAL_MODE = False                # False = treino, True = avaliação

# Para modo AVALIAÇÃO: especifique a run que quer testar
DEFAULT_RUN_NAME = None                  # Ex: "B_Base_esmc_300m_20241218_143022" ou None para mais recente
```

**Exemplos de uso:**

```bash
# 1. Treino com defaults
python main.py  # Usa os defaults configurados

# 2. Avaliação automática (mais recente)
# DEFAULT_EVAL_MODE = True
# DEFAULT_RUN_NAME = None  
python main.py  # Usa o modelo mais recente

# 3. Avaliação de modelo específico
# DEFAULT_EVAL_MODE = True
# DEFAULT_RUN_NAME = "B_Base_esmc_300m_20241218_143022"
python main.py  # Usa exatamente este modelo
```

### Treinamento

```bash
# Exemplo básico - dataset B com modelo ESMC 300M
python main.py --dataset B --model esmc_300m --virus-type Tudo

# Usar arquivos base (sem sufixo) - simB.txt, naoB.txt
python main.py --dataset B --model esmc_300m --virus-type Base

# Treinar MHC1 com ESM2
python main.py --dataset MHC1 --model esm2_t33_650M_UR50D --virus-type Virus

# Treinar MHC2 com modelo maior
python main.py --dataset MHC2 --model esm2_t36_3B_UR50D --virus-type Lent

# Treino rápido (poucas épocas)
python main.py --dataset B --model esmc_300m --virus-type Base --epochs 5
```

### Avaliação

```bash
# Avaliar melhor modelo treinado (carrega automaticamente o best_model.pt)
python main.py --eval --dataset B --model esmc_300m --virus-type Tudo

# Avaliar checkpoint específico
python main.py --eval --dataset MHC1 --model esm2_t33_650M_UR50D --virus-type Virus --step 1500
```

### Parâmetros Disponíveis

#### 🆕 CONTROLE TOTAL dos Hiperparâmetros

**Configuração Básica:**
- `--dataset`: Escolha entre `B`, `MHC1`, `MHC2` (default: configurável)
- `--model`: Tipo de modelo a usar (default: configurável)
  - `esmc_300m`: ESM-C 300M parâmetros
  - `esmc_600m`: ESM-C 600M parâmetros  
  - `esm2_t33_650M_UR50D`: ESM2 650M parâmetros
  - `esm2_t36_3B_UR50D`: ESM2 3B parâmetros
- `--virus-type`: Tipo de dados de vírus (default: configurável)
  - `Base`: Arquivos base sem sufixo (ex: `simB.txt`, `naoB.txt`)
  - `Tudo`: Todos os dados (ex: `simBTudo.txt`, `naoBTudo.txt`)
  - `Virus`: Apenas dados de vírus (ex: `simBVirus.txt`, `naoBVirus.txt`)
  - `Lent`: Apenas lentivírus (ex: `simBLent.txt`, `naoBLent.txt`)
  - `Retro`: Apenas retrovírus (ex: `simBRetro.txt`, `naoBRetro.txt`)

**🆕 Hiperparâmetros de Treinamento:**
- `--epochs`: Número de épocas de treinamento (default: configurável)
- `--lr`, `--learning-rate`: Taxa de aprendizado (default: configurável)
- `--weight-decay`: Regularização L2 (default: configurável)
- `--batch-size`: Tamanho do batch (default: auto baseado no modelo)
- `--max-length`: Comprimento máximo das sequências (default: auto baseado no modelo)

**🆕 Configuração do Modelo:**
- `--dropout`: Taxa de dropout (default: auto baseado no modelo)
- `--freeze-backbone`: Congelar pesos do encoder (default: configurável)

**🆕 Pesos da Loss Function:**
- `--pos-class-weight`: Peso para classe negativa para melhorar precision (default: configurável)
- `--loss-weight-multiplier`: Multiplicador escalar para os pesos (default: configurável)

**🆕 Intervalos e Salvamento:**
- `--eval-interval`: Avaliar a cada N épocas (default: configurável)
- `--save-interval`: Salvar checkpoint a cada N épocas (default: configurável)

**🆕 Weights & Biases:**
- `--wandb-project`: Nome do projeto W&B (default: auto gerado)
- `--wandb-entity`: Organização W&B (default: conta padrão)

**🆕 Reprodutibilidade:**
- `--seed`: Semente para reprodutibilidade (default: configurável)

**Avaliação:**
- `--eval`: Modo de avaliação (default: False = treinamento)
- `--step`: Step específico para avaliação (default: melhor modelo)
- `--run-name`: Run específico para avaliar (default: mais recente)
- `--list-runs`: Listar runs disponíveis

## ⚙️ Configurações Automáticas

O sistema configura automaticamente os hiperparâmetros baseado no modelo escolhido:

### ESMC Models
- **Learning Rate**: 5e-3 (maior para ESMC)
- **Max Length**: 30 (otimizado para peptídeos)
- **Batch Sizes**: 8 (300M), 4 (600M)

### ESM2 Models  
- **Learning Rate**: 1e-5 (menor para ESM2)
- **Max Length**: 30 (otimizado para peptídeos)
- **Batch Sizes**: 6 (650M), 2 (3B)

## 🎯 Controle de Precision com Pesos Personalizados

O sistema oferece controle fino sobre a loss function para otimizar a métrica de precision através de pesos nas classes:

### Parâmetros de Peso

- **`pos_class_weight`** (default: 1.0): Peso aplicado à classe negativa para controle de precision
  - `> 1.0`: Penaliza mais falsos positivos → **melhora precision**
  - `< 1.0`: Penaliza menos falsos positivos → melhora recall
  - `= 1.0`: Pesos balanceados (comportamento padrão)

- **`loss_weight_multiplier`** (default: 1.0): Multiplicador escalar para amplificar o efeito
  - Multiplica ambos os pesos das classes
  - Útil para ajuste fino adicional

### Fórmula dos Pesos (CORRIGIDA)
```
peso_classe_negativa = pos_class_weight * loss_weight_multiplier
peso_classe_positiva = 1.0 * loss_weight_multiplier
```

### Exemplos de Configuração

**Peso moderado para melhorar precision:**
```python
trainer = Trainer(
    # ... outros parâmetros
    pos_class_weight=2.0,           # Classe positiva 2x mais penalizada
    loss_weight_multiplier=1.0,     # Sem amplificação adicional
)
# Resultado: pesos [1.0, 2.0]
```

**Peso alto para precision muito conservadora:**
```python
trainer = Trainer(
    # ... outros parâmetros  
    pos_class_weight=5.0,           # Classe positiva 5x mais penalizada
    loss_weight_multiplier=1.5,     # Amplifica o efeito em 1.5x
)
# Resultado: pesos [1.5, 7.5]
```

**Para melhorar recall (menos conservador):**
```python
trainer = Trainer(
    # ... outros parâmetros
    pos_class_weight=0.5,           # Classe positiva menos penalizada
    loss_weight_multiplier=2.0,     # Amplifica o efeito
)
# Resultado: pesos [2.0, 1.0]
```

### Recomendações de Uso

- **Precision baixa**: Use `pos_class_weight=2.0` ou maior
- **Recall baixo**: Use `pos_class_weight=0.5` ou menor  
- **Dataset desbalanceado**: Ajuste baseado na distribuição real
- **Ajuste fino**: Use `loss_weight_multiplier` para amplificar sutilmente

### Monitoramento
Durante o treinamento, o sistema exibe os pesos configurados:
```
🔧 Loss function configurada:
   Peso classe negativa (0): 1.000
   Peso classe positiva (1): 2.000
   Multiplicador: 1.000
   💡 Peso maior na classe positiva → menos falsos positivos → melhor precision
```

## 🎯 Critério de Melhor Modelo

O sistema usa um critério inteligente para selecionar o melhor modelo:

- **Métrica Combinada**: F1 + Precision (média harmônica)
- **Objetivo**: Modelos com **F1 alto E Precision alto**
- **Evita**: Modelos com recall alto mas precision baixa
- **Formula**: `2 * (F1 * Precision) / (F1 + Precision)`

Isso garante que o melhor modelo tenha boa performance geral (F1) sem sacrificar a confiabilidade das predições positivas (Precision).

## 📊 Conjuntos de Dados

### Treinamento
- **Fonte**: Arquivos nas pastas B/, MHC1/, MHC2/
- **100% dos dados**: Todos os dados disponíveis para treino
- **Arquivos**: Baseados no `--virus-type` escolhido
- **Sem validação**: Modelo treina em todos os dados disponíveis

### Teste
- **Fonte**: Arquivos específicos em `../Inferencia_HIV/`
- **Independente**: Dados HIV específicos para cada dataset
- **Fixo**: Sempre os mesmos arquivos para comparabilidade
- **Caminho**: Inferencia_HIV está no mesmo nível que Modelos/

## 📈 Outputs

### Estrutura de Saída
```
{dataset}/model/{run_name}/
├── best_model.pt               # Melhor modelo baseado no critério
├── best_model_info.json       # Informações do melhor modelo
├── pytorch_model_step{N}.pt    # Checkpoints regulares
├── model_metrics.json         # Histórico completo de métricas
└── run_info.json              # Informações do treinamento
```

### Métricas Computadas
- **Accuracy**: Acurácia geral
- **Precision**: Precisão (PPV) - **usado no critério**
- **Recall**: Sensibilidade (TPR)
- **Specificity**: Especificidade (TNR)
- **F1**: F1-score - **usado no critério**
- **MCC**: Matthews Correlation Coefficient
- **AUC**: Area Under the ROC Curve
- **F1_Precision_Combined**: Métrica combinada para seleção

## 🔧 Exemplos de Uso Avançado

### 1. Uso com defaults (recomendado para testes rápidos)
```bash
# Edite os defaults no main.py uma vez
# DEFAULT_DATASET = "MHC2"
# DEFAULT_MODEL = "esmc_300m" 
# DEFAULT_VIRUS_TYPE = "Base"
# DEFAULT_EPOCHS = 10

# Depois simplesmente execute:
python main.py                    # Treino rápido com defaults
python main.py --eval             # Avaliação com defaults
```

### 1.1. Workflow completo com defaults
```bash
# 1. Primeiro treinar alguns modelos
python main.py --epochs 5         # Treino rápido
python main.py --epochs 15        # Treino médio  
python main.py --epochs 30        # Treino completo

# 2. Ver modelos disponíveis
python main.py --list-runs
# Output: Lista com B_Base_esmc_300m_TIMESTAMP para cada treino

# 3. Configurar default para avaliar modelo específico
# Edite no main.py:
# DEFAULT_EVAL_MODE = True
# DEFAULT_RUN_NAME = "B_Base_esmc_300m_20241218_143055"  # Copie da lista

# 4. Avaliar sem argumentos
python main.py                    # Avalia exatamente o modelo configurado!
```

### 2. Comparar diferentes tipos de arquivos no mesmo dataset
```bash
# Comparar arquivo base vs todos os dados
python main.py --dataset B --model esmc_300m --virus-type Base  # simB.txt
python main.py --dataset B --model esmc_300m --virus-type Tudo  # simBTudo.txt
```

### 3. Comparar modelos no mesmo dataset
```bash
# Treinar todos os modelos no dataset B com arquivos base
for model in esmc_300m esmc_600m esm2_t33_650M_UR50D; do
    python main.py --dataset B --model $model --virus-type Base
done
```

### 4. Treinar em todos os datasets
```bash
# Treinar ESMC 300M em todos os datasets com arquivos base
for dataset in B MHC1 MHC2; do
    python main.py --dataset $dataset --model esmc_300m --virus-type Base
done
```

### 5. Experimentos com diferentes tipos de vírus
```bash
# Testar todos os tipos de vírus no MHC1
for virus in Base Tudo Virus Lent Retro; do
    python main.py --dataset MHC1 --model esmc_300m --virus-type $virus --epochs 10
done
```

### 6. Avaliar todos os melhores modelos
```bash
# Avaliar os melhores modelos treinados
for dataset in B MHC1 MHC2; do
    python main.py --eval --dataset $dataset --model esmc_300m --virus-type Base
done
```

### 7. Treinamento rápido para testes
```bash
# Treinos rápidos (5 épocas) para teste
python main.py --dataset B --model esmc_300m --virus-type Base --epochs 5
python main.py --dataset MHC1 --model esmc_300m --virus-type Virus --epochs 5
python main.py --dataset MHC2 --model esmc_300m --virus-type Lent --epochs 5
```

## 📝 Logs e Monitoramento

- **Weights & Biases**: Logs automáticos de treinamento e métricas
- **Project Names**: Automático (`protein-b`, `protein-mhc1`, `protein-mhc2`)
- **Run Names**: Formato `{dataset}_{virus_type}_{model_type}`
- **Métricas locais**: Salvos em JSON para análise offline
- **Progresso visual**: Métricas principais mostradas a cada época

## 🏆 Sistema de Melhor Modelo

### Durante o Treinamento
1. A cada época, calcula métricas no conjunto de teste
2. Computa score combinado F1+Precision
3. Se melhor que anterior, salva como `best_model.pt`
4. Mantém histórico completo em `best_model_info.json`

### Na Avaliação
1. Por padrão, carrega `best_model.pt`
2. Opção de carregar checkpoint específico com `--step`
3. Mostra métricas detalhadas e comparativas

## 🚨 Requisitos

- Python 3.8+
- PyTorch
- Transformers
- ESM (para modelos ESMC)
- Weights & Biases
- scikit-learn
- tqdm
- tabulate

## 💡 Dicas

1. **Melhor modelo**: O sistema sempre salva o melhor baseado no critério F1+Precision
2. **Conjuntos de teste**: São específicos para HIV e independentes do treinamento
3. **Sem validação**: Treina em 100% dos dados disponíveis, testa apenas no conjunto HIV
4. **Avaliação**: Use `--eval` para testar modelos já treinados
5. **Logs**: Acompanhe o treinamento via Weights & Biases ou outputs locais
6. **Caminho correto**: Inferencia_HIV está no mesmo nível que a pasta Modelos/ 