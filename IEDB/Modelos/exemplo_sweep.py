#!/usr/bin/env python3
"""
🎯 EXEMPLO PRÁTICO: Como usar o W&B Hyperparameter Sweep

Este arquivo mostra exemplos práticos de como usar o sweep para otimizar precision.
"""

# ================================================================== #
# 💡 EXEMPLOS DE USO DO SWEEP                                        #
# ================================================================== #

print("""
🚀 EXEMPLO 1: Sweep Rápido para Testes (10 runs)
────────────────────────────────────────────────

# 1. Editar sweep_config.yaml:
run_cap: 10

# 2. Configurar defaults no main.py:
DEFAULT_DATASET = "B"
DEFAULT_MODEL = "esmc_300m"
DEFAULT_EPOCHS = 3
DEFAULT_SWEEP_MODE = True

# 3. Executar:
python main.py

════════════════════════════════════════════════════════════════════
""")

print("""
🔧 EXEMPLO 2: Sweep Completo para Otimização (50 runs)
──────────────────────────────────────────────────────

# 1. Manter sweep_config.yaml padrão (run_cap: 50)

# 2. Configurar para dataset específico:
python main.py --sweep \\
    --dataset MHC1 \\
    --model esmc_600m \\
    --virus-type Base \\
    --epochs 5 \\
    --wandb-project "precision-optimization-mhc1"

════════════════════════════════════════════════════════════════════
""")

print("""
📊 EXEMPLO 3: Analisando Resultados do Sweep
─────────────────────────────────────────────

Após o sweep, você verá no W&B Dashboard:

MELHORES HIPERPARÂMETROS encontrados:
├── pos_class_weight: 3.7        # Peso ótimo para precision
├── loss_weight_multiplier: 1.8  # Multiplicador ótimo
├── lr: 2.4e-4                   # Learning rate ótimo
└── weight_decay: 0.02           # Regularização ótima

MÉTRICAS ALCANÇADAS:
├── f1_precision_combined: 0.87  # Métrica otimizada
├── precision: 0.91              # Precision melhorada!
├── f1: 0.84                     # F1 mantido alto
└── recall: 0.78                 # Recall pode diminuir

════════════════════════════════════════════════════════════════════
""")

print("""
🎯 EXEMPLO 4: Aplicando os Melhores Hiperparâmetros
───────────────────────────────────────────────────

# Após identificar os melhores hiperparâmetros no sweep:

python main.py \\
    --dataset MHC1 \\
    --model esmc_600m \\
    --virus-type Base \\
    --epochs 30 \\
    --pos-class-weight 3.7 \\
    --loss-weight-multiplier 1.8 \\
    --lr 2.4e-4 \\
    --weight-decay 0.02

# Resultado: Modelo final otimizado com os melhores hiperparâmetros!

════════════════════════════════════════════════════════════════════
""")

print("""
🔄 EXEMPLO 5: Workflow Completo de Otimização
─────────────────────────────────────────────

1. 🧪 SWEEP RÁPIDO (testar configuração):
   python main.py --sweep --epochs 3 --wandb-project "test-sweep"

2. 📊 SWEEP COMPLETO (otimização real):
   python main.py --sweep --epochs 5 --wandb-project "precision-opt"

3. 🎯 TREINO FINAL (usar melhores hiperparâmetros):
   python main.py --epochs 30 --pos-class-weight X.X --lr X.Xe-4

4. 🔍 AVALIAÇÃO:
   python main.py --eval

════════════════════════════════════════════════════════════════════
""")

print("""
⚙️ CONFIGURAÇÃO PERSONALIZADA DO SWEEP
──────────────────────────────────────

Para focar AINDA MAIS em precision, edite sweep_config.yaml:

# Pesos mais altos para precision extrema
parameters:
  pos_class_weight:
    min: 3.0      # Começar já com peso alto
    max: 20.0     # Testar pesos muito altos
  
  loss_weight_multiplier:
    min: 1.5      # Amplificação mínima
    max: 10.0     # Amplificação máxima

# Resultado esperado: Precision muito alta, recall pode diminuir

════════════════════════════════════════════════════════════════════
""")

print("""
🏆 MELHORES PRÁTICAS PARA SWEEP
──────────────────────────────

✅ DO:
• Começar com sweep rápido (10 runs, 3 épocas)
• Usar sweep completo apenas quando necessário
• Acompanhar progresso no W&B Dashboard
• Testar hiperparâmetros encontrados em treino final
• Fazer backup dos melhores hiperparâmetros

❌ DON'T:
• Executar sweep com muitas épocas (desperdiça tempo)
• Ignorar o early termination (deixa runs ruins correrem)
• Modificar batch_size/max_length no sweep (instabilidade)
• Esquecer de fazer wandb login
• Executar múltiplos sweeps simultaneamente

════════════════════════════════════════════════════════════════════
""")

# ================================================================== #
# 📈 MÉTRICAS ESPERADAS DO SWEEP                                     #
# ================================================================== #

metricas_esperadas = {
    "Baseline (sem otimização)": {
        "precision": 0.65,
        "recall": 0.85,
        "f1": 0.73,
        "f1_precision_combined": 0.69
    },
    "Após Sweep (otimizado)": {
        "precision": 0.82,  # ⬆️ Melhoria significativa
        "recall": 0.78,     # ⬇️ Pequena redução aceitável
        "f1": 0.80,         # ➡️ Mantido/melhorado
        "f1_precision_combined": 0.81  # ⬆️ Objetivo principal
    }
}

print("📈 MÉTRICAS ESPERADAS:")
print("═" * 60)

for cenario, metricas in metricas_esperadas.items():
    print(f"\n{cenario}:")
    for metrica, valor in metricas.items():
        print(f"   {metrica}: {valor:.2f}")

print("\n" + "═" * 60)
print("💡 O sweep foca em maximizar f1_precision_combined!")
print("🎯 Resultado: Precision alta SEM sacrificar muito F1!")