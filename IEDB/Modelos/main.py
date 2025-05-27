import argparse
import os
from pathlib import Path
from config import get_config
from trainer import Trainer
import wandb
import yaml

# ================================================================== #
# 🔧 CONFIGURAÇÃO DE DEFAULTS (EDITE AQUI PARA FACILIZAR O USO)     #
# ================================================================== #
# Edite estes valores para evitar usar argumentos na linha de comando

# ────── Configuração básica ──────────────────────────────────────
DEFAULT_DATASET = "MHC2"                    # Opções: "B", "MHC1", "MHC2"
DEFAULT_MODEL = "esmc_600m"                  # Opções: "esmc_300m", "esmc_600m", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"
DEFAULT_VIRUS_TYPE = "Base"                  # Opções: "Base", "Tudo", "Virus", "Lent", "Retro"
DEFAULT_EVAL_MODE = False                    # True = apenas avaliação, False = treinamento

# ────── Hiperparâmetros de treinamento ───────────────────────────
DEFAULT_EPOCHS = 5                         # Número de épocas (max_iter para early terminate)
DEFAULT_LEARNING_RATE = 5e-5              # Taxa de aprendizado
DEFAULT_WEIGHT_DECAY = 0.00                 # Decaimento de peso (L2 regularization)
DEFAULT_BATCH_SIZE = 128                   # None = usar padrão do modelo, ou especificar valor
DEFAULT_MAX_LENGTH = None                   # None = usar padrão do modelo, ou especificar valor

# ────── Configuração do modelo ───────────────────────────────────
DEFAULT_DROPOUT = 0.0                      # None = usar padrão do modelo, ou especificar (ex: 0.3)
DEFAULT_FREEZE_BACKBONE = False             # True = congelar encoder, False = treinar tudo

# ────── Pesos da loss function para precision ───────────────────
DEFAULT_POS_CLASS_WEIGHT = 3.0             # Peso aplicado à classe negativa (> 1.0 melhora precision)
DEFAULT_LOSS_WEIGHT_MULTIPLIER = 1.0       # Multiplicador escalar adicional

# ────── Para modo AVALIAÇÃO ──────────────────────────────────────
DEFAULT_RUN_NAME = None                     # Ex: "B_Base_esmc_300m_20241218_143022" ou None para mais recente

# ────── Intervalos de salvamento ─────────────────────────────────
DEFAULT_EVAL_INTERVAL = 1                  # Avaliar a cada N épocas
DEFAULT_SAVE_INTERVAL = 1                  # Salvar checkpoint a cada N épocas

# ────── Configuração W&B ─────────────────────────────────────────
DEFAULT_WANDB_PROJECT = None               # None = automático (protein-{dataset}), ou especificar
DEFAULT_WANDB_ENTITY = None                # None = conta padrão, ou especificar organização

# ────── Reprodutibilidade ────────────────────────────────────────
DEFAULT_SEED = 42                          # Semente para reprodutibilidade

# ────── W&B Hyperparameter Sweep ─────────────────────────────────
DEFAULT_SWEEP_MODE = False                 # True = ativar sweep para otimização automática

# ────── Pré-treinamento MLM ──────────────────────────────────────
DEFAULT_PRETRAIN_MODE = False              # True = executar pré-treinamento MLM
DEFAULT_PRETRAIN_EPOCHS = 15               # Épocas de pré-treinamento
DEFAULT_PRETRAIN_LR = 5e-4                 # Learning rate para pré-treinamento
DEFAULT_PRETRAIN_BATCH_SIZE = 128           # Batch size para pré-treinamento
DEFAULT_PRETRAIN_MAX_LENGTH = 60          # Comprimento máximo para epitopos
DEFAULT_MLM_PROBABILITY = 0.15             # Probabilidade de mascaramento MLM
DEFAULT_PRETRAINED_BACKBONE_PATH = "/home/ubuntu/guilherme.evangelista/HIV-Models/IEDB/Modelos/MHC2/model/pretraining/pretrain_esmc_600m_20250527_031230/best_pretrained_esmc.pt"   # Caminho para backbone pré-treinado
DEFAULT_RESUME_PRETRAINING_BACKBONE_PATH = None

# ================================================================== #


def list_available_runs(dataset, model, virus_type):
    """Lista runs disponíveis para avaliação"""
    base_path = Path(__file__).parent
    model_dir = base_path / dataset / "model"
    
    if not model_dir.exists():
        print(f"❌ Pasta de modelos não encontrada: {model_dir}")
        return []
    
    # Padrão: {dataset}_{virus_type}_{model}_{timestamp}
    pattern = f"{dataset}_{virus_type}_{model}_*"
    runs = list(model_dir.glob(pattern))
    
    if not runs:
        print(f"❌ Nenhum modelo encontrado para {dataset}_{virus_type}_{model}")
        return []
    
    # Ordenar por timestamp (mais recente primeiro)
    runs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    
    print(f"\n📂 Runs disponíveis para {dataset}_{virus_type}_{model}:")
    for i, run in enumerate(runs):
        timestamp = run.name.split('_')[-1]
        best_model = run / "best_model.pt"
        status = "✅ best_model.pt" if best_model.exists() else "⚠️  sem best_model.pt"
        print(f"   {i+1}. {run.name} ({status})")
    
    return runs


def parse_args():
    parser = argparse.ArgumentParser(description="Training or Evaluation mode for HIV protein models")
    
    # ────── Modo de execução ─────────────────────────────────────────
    parser.add_argument(
        "--eval", action="store_true", default=DEFAULT_EVAL_MODE,
        help="Run in evaluation mode (default: False)"
    )
    
    # ────── Configuração básica ──────────────────────────────────────
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, choices=["B", "MHC1", "MHC2"],
        help=f"Dataset to use: B, MHC1, or MHC2 (default: {DEFAULT_DATASET})"
    )
    
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=["esmc_300m", "esmc_600m", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"],
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--virus-type", type=str, default=DEFAULT_VIRUS_TYPE, 
        choices=["Base", "Tudo", "Virus", "Lent", "Retro"],
        help=f"Type of virus data to use (default: {DEFAULT_VIRUS_TYPE}). 'Base' uses files without suffix"
    )
    
    # ────── Hiperparâmetros de treinamento ───────────────────────────
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help=f"Weight decay for L2 regularization (default: {DEFAULT_WEIGHT_DECAY})"
    )
    
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: auto based on model)"
    )
    
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH,
        help=f"Maximum sequence length (default: auto based on model)"
    )
    
    # ────── Configuração do modelo ───────────────────────────────────
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_DROPOUT,
        help=f"Dropout rate (default: auto based on model)"
    )
    
    parser.add_argument(
        "--freeze-backbone", action="store_true", default=DEFAULT_FREEZE_BACKBONE,
        help=f"Freeze backbone weights (default: {DEFAULT_FREEZE_BACKBONE})"
    )
    
    # ────── Pesos da loss function ───────────────────────────────────
    parser.add_argument(
        "--pos-class-weight", type=float, default=DEFAULT_POS_CLASS_WEIGHT,
        help=f"Weight for negative class to improve precision (default: {DEFAULT_POS_CLASS_WEIGHT})"
    )
    
    parser.add_argument(
        "--loss-weight-multiplier", type=float, default=DEFAULT_LOSS_WEIGHT_MULTIPLIER,
        help=f"Multiplier for loss weights (default: {DEFAULT_LOSS_WEIGHT_MULTIPLIER})"
    )
    
    # ────── Intervalos e salvamento ──────────────────────────────────
    parser.add_argument(
        "--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL,
        help=f"Evaluate every N epochs (default: {DEFAULT_EVAL_INTERVAL})"
    )
    
    parser.add_argument(
        "--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL,
        help=f"Save checkpoint every N epochs (default: {DEFAULT_SAVE_INTERVAL})"
    )
    
    # ────── Weights & Biases ─────────────────────────────────────────
    parser.add_argument(
        "--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name (default: auto generated)"
    )
    
    parser.add_argument(
        "--wandb-entity", type=str, default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity/organization (default: None)"
    )
    
    # ────── Reprodutibilidade ────────────────────────────────────────
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    
    # ────── W&B Hyperparameter Sweep ─────────────────────────────────
    parser.add_argument(
        "--sweep", action="store_true", default=DEFAULT_SWEEP_MODE,
        help=f"Enable W&B hyperparameter sweep (default: {DEFAULT_SWEEP_MODE})"
    )
    
    # ────── Pré-treinamento MLM ──────────────────────────────────────
    parser.add_argument(
        "--pretrain", action="store_true", default=DEFAULT_PRETRAIN_MODE,
        help=f"Run MLM pretraining before fine-tuning (default: {DEFAULT_PRETRAIN_MODE})"
    )
    
    parser.add_argument(
        "--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS,
        help=f"Total number of pretraining epochs desired (default: {DEFAULT_PRETRAIN_EPOCHS}). When resuming, this is the final epoch number."
    )
    
    parser.add_argument(
        "--pretrain-lr", type=float, default=DEFAULT_PRETRAIN_LR,
        help=f"Learning rate for pretraining (default: {DEFAULT_PRETRAIN_LR})"
    )
    
    parser.add_argument(
        "--pretrain-batch-size", type=int, default=DEFAULT_PRETRAIN_BATCH_SIZE,
        help=f"Batch size for pretraining (default: {DEFAULT_PRETRAIN_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "--pretrain-max-length", type=int, default=DEFAULT_PRETRAIN_MAX_LENGTH,
        help=f"Max length for pretraining epitopes (default: {DEFAULT_PRETRAIN_MAX_LENGTH})"
    )
    
    parser.add_argument(
        "--mlm-probability", type=float, default=DEFAULT_MLM_PROBABILITY,
        help=f"MLM masking probability (default: {DEFAULT_MLM_PROBABILITY})"
    )
    
    parser.add_argument(
        "--pretrained-backbone-path", type=str, default=DEFAULT_PRETRAINED_BACKBONE_PATH,
        help=f"Path to pretrained backbone weights (default: None)"
    )
    
    parser.add_argument(
        "--resume-pretraining", type=str, default=DEFAULT_RESUME_PRETRAINING_BACKBONE_PATH,
        help="Path to checkpoint to resume pretraining from"
    )
    
    # ────── Avaliação específica ─────────────────────────────────────
    parser.add_argument(
        "--step", type=int, default=None,
        help="Specific step to load for evaluation (default: melhor modelo)"
    )
    
    parser.add_argument(
        "--run-name", type=str, default=DEFAULT_RUN_NAME,
        help=f"Specific run name to evaluate (default: {DEFAULT_RUN_NAME or 'most recent'}). Use --list-runs to see available runs"
    )
    
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List available runs for the specified dataset/model/virus-type and exit"
    )
    
    parser.add_argument(
        "--list-pretraining", action="store_true",
        help="List available pretraining checkpoints and backbones for the specified dataset/model and exit"
    )
    
    return parser.parse_args()


def sweep_train():
    """
    Função para treinamento durante o sweep do W&B.
    Esta função será chamada pelo sweep agent para cada combinação de hiperparâmetros.
    """
    with wandb.init() as run:
        # Pegar hiperparâmetros sugeridos pelo sweep
        config_wandb = wandb.config
        
        # Usar defaults do main.py como base
        args = parse_args()
        
        # Override com hiperparâmetros do sweep
        if hasattr(config_wandb, 'pos_class_weight'):
            args.pos_class_weight = config_wandb.pos_class_weight
        if hasattr(config_wandb, 'loss_weight_multiplier'):
            args.loss_weight_multiplier = config_wandb.loss_weight_multiplier
        if hasattr(config_wandb, 'lr'):
            args.lr = config_wandb.lr
        if hasattr(config_wandb, 'weight_decay'):
            args.weight_decay = config_wandb.weight_decay
        if hasattr(config_wandb, 'dropout'):
            args.dropout = config_wandb.dropout
        
        # Get configuration básica
        config = get_config(
            dataset_type=args.dataset,
            model_type=args.model,
            virus_type=args.virus_type
        )
        
        # Apply hiperparâmetros
        config["eval"] = False  # Sempre treinamento no sweep
        
        # Usar épocas fixas do args - W&B controlará early terminate externamente
        config["epochs"] = args.epochs
            
        config["seed"] = args.seed
        config["lr"] = args.lr
        config["weight_decay"] = args.weight_decay
        config["pos_class_weight"] = args.pos_class_weight
        config["loss_weight_multiplier"] = args.loss_weight_multiplier
        config["eval_interval"] = args.eval_interval
        config["save_interval"] = args.save_interval
        
        # Override configurações específicas para sweep
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.max_length is not None:
            config["max_length"] = args.max_length
        if args.dropout is not None:
            config["dropout"] = args.dropout
        config["freeze_backbone"] = args.freeze_backbone
        
        # W&B config
        if args.wandb_project is not None:
            config["project"] = args.wandb_project
        config["entity"] = args.wandb_entity
        
        # NOVO: Adicionar pretrained_backbone_path ao sweep
        if args.pretrained_backbone_path:
            config["pretrained_backbone_path"] = args.pretrained_backbone_path
        
        # Gerar run_name específico para sweep
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "manual"
        run_id = wandb.run.id
        config["run_name"] = f"sweep_{sweep_id}_{run_id}"
        
        print(f"\n🎯 SWEEP RUN - Hiperparâmetros testados:")
        print(f"{'='*50}")
        print(f"   Pos Class Weight: {args.pos_class_weight:.3f}")
        print(f"   Loss Weight Multiplier: {args.loss_weight_multiplier:.3f}")
        print(f"   Learning Rate: {args.lr:.6f}")
        print(f"   Weight Decay: {args.weight_decay:.6f}")
        print(f"   Dropout: {config.get('dropout', 'auto')}")
        print(f"   Max Épocas: {config['epochs']} (W&B pode parar antes)")
        
        # Mostrar se está usando backbone pré-treinado
        if args.pretrained_backbone_path:
            backbone_name = Path(args.pretrained_backbone_path).name
            print(f"   🧬 Backbone pré-treinado: {backbone_name}")
        else:
            print(f"   🧬 Backbone: Original (sem pré-treinamento)")
        
        print(f"{'='*50}")
        
        # Executar treinamento
        trainer = Trainer(**config)
        trainer.run()


def start_sweep():
    """
    Inicia um novo sweep do W&B.
    """
    # Carregar configuração do sweep
    sweep_config_path = Path(__file__).parent / "sweep_config.yaml"
    
    if not sweep_config_path.exists():
        print(f"❌ Arquivo de configuração do sweep não encontrado: {sweep_config_path}")
        print("💡 Certifique-se de que o arquivo sweep_config.yaml existe")
        return
    
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Obter configurações do usuário
    args = parse_args()
    
    # Determinar nome do projeto W&B
    project_name = args.wandb_project if args.wandb_project else f"protein-{args.dataset.lower()}-sweep"
    
    print(f"\n🚀 INICIANDO W&B SWEEP")
    print(f"{'='*60}")
    print(f"📁 Dataset: {args.dataset}")
    print(f"🧠 Modelo: {args.model}")
    print(f"🦠 Tipo vírus: {args.virus_type}")
    print(f"📊 Projeto W&B: {project_name}")
    print(f"🎯 Métrica: {sweep_config['metric']['name']}")
    print(f"🔄 Max runs: {sweep_config.get('run_cap', 'unlimited')}")
    print(f"{'='*60}")
    
    # Criar sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=project_name,
        entity=args.wandb_entity
    )
    
    print(f"\n✅ Sweep criado com ID: {sweep_id}")
    print(f"🌐 Acompanhe em: https://wandb.ai/{args.wandb_entity or 'sua-conta'}/{project_name}/sweeps/{sweep_id}")
    print(f"\n🤖 Iniciando sweep agent...")
    
    # Executar sweep
    wandb.agent(sweep_id, sweep_train, count=sweep_config.get('run_cap', None))
    
    print(f"\n🏁 Sweep concluído!")


def main():
    args = parse_args()
    
    # ────── Modo Sweep ────────────────────────────────────────────────
    if args.sweep:
        start_sweep()
        return
    
    # Se pediu para listar runs, faz isso e sai
    if args.list_runs:
        list_available_runs(args.dataset, args.model, args.virus_type)
        return
    
    # Se pediu para listar pré-treinamentos, faz isso e sai
    if args.list_pretraining:
        from pretrainer import list_pretraining_checkpoints
        list_pretraining_checkpoints(args.dataset, args.model)
        return
    
    # Get configuration with specified parameters (apenas caminhos e estruturas básicas)
    config = get_config(
        dataset_type=args.dataset,
        model_type=args.model,
        virus_type=args.virus_type
    )

    # Se está em modo avaliação e especificou um run
    if args.eval and args.run_name:
        # Usar o run específico (seja do default ou do argumento)
        config["run_name"] = args.run_name
        print(f"🎯 Usando run específico: {args.run_name}")
    elif args.eval:
        # Modo avaliação sem run específico - buscar o mais recente
        runs = list_available_runs(args.dataset, args.model, args.virus_type)
        if not runs:
            print("❌ Nenhum modelo treinado encontrado!")
            print("💡 Dica: Primeiro treine um modelo com: python main.py")
            return
        
        # Usar o mais recente
        config["run_name"] = runs[0].name
        print(f"🎯 Usando run mais recente: {runs[0].name}")

    # ────── Override config com hiperparâmetros do main.py ───────────
    # Básicos
    config["eval"] = args.eval
    config["step"] = args.step
    config["epochs"] = args.epochs
    config["seed"] = args.seed
    
    # Hiperparâmetros de treinamento
    config["lr"] = args.lr
    config["weight_decay"] = args.weight_decay
    
    # Override batch_size e max_length se especificados
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.max_length is not None:
        config["max_length"] = args.max_length
    
    # Configuração do modelo
    if args.dropout is not None:
        config["dropout"] = args.dropout
    config["freeze_backbone"] = args.freeze_backbone
    
    # Pesos da loss function
    config["pos_class_weight"] = args.pos_class_weight
    config["loss_weight_multiplier"] = args.loss_weight_multiplier
    
    # Intervalos
    config["eval_interval"] = args.eval_interval
    config["save_interval"] = args.save_interval
    
    # Weights & Biases
    if args.wandb_project is not None:
        config["project"] = args.wandb_project
    config["entity"] = args.wandb_entity
    
    # ────── Pré-treinamento MLM (se habilitado) ──────────────────────
    pretrained_backbone_path = args.pretrained_backbone_path
    
    if (args.pretrain or args.resume_pretraining) and not args.eval and args.model.startswith("esmc"):
        if args.resume_pretraining:
            print(f"\n🔄 CONTINUANDO PRÉ-TREINAMENTO MLM")
        else:
            print(f"\n🧬 EXECUTANDO PRÉ-TREINAMENTO MLM")
        print(f"{'='*60}")
        
        from pretrainer import PretrainerESMC, load_sequences_from_files
        
        # Carregar sequências de treino para pré-treinamento
        sequences = load_sequences_from_files(config["train_pos"], config["train_neg"])
        
        if len(sequences) < 100:
            print(f"⚠️  Poucas sequências para pré-treinamento ({len(sequences)})")
            print(f"💡 Recomendamos pelo menos 1000 sequências para bons resultados")
        
        # Configurar pré-treinador
        pretrainer_config = {
            "sequences": sequences,
            "artifacts_path": config["artifacts_path"],
            "lr": args.pretrain_lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.pretrain_batch_size,
            "max_length": args.pretrain_max_length,
            "epochs": args.pretrain_epochs,
            "base_model": args.model,
            "mlm_probability": args.mlm_probability,
            "project": f"esmc-pretrain-{args.dataset.lower()}",
            "entity": args.wandb_entity,
            "seed": args.seed,
            "resume_from_checkpoint": args.resume_pretraining,
        }
        
        if args.resume_pretraining:
            print(f"🔄 Continuando pré-treinamento de:")
            print(f"   📁 {args.resume_pretraining}")
        else:
            print(f"🚀 Iniciando pré-treinamento:")
        
        print(f"   📊 {len(sequences)} sequências")
        print(f"   🕐 {args.pretrain_epochs} épocas")
        print(f"   📏 Max length: {args.pretrain_max_length}")
        print(f"   🎭 MLM prob: {args.mlm_probability}")
        print(f"   📚 Batch size: {args.pretrain_batch_size}")
        print(f"   🎯 Learning rate: {args.pretrain_lr}")
        
        # Executar pré-treinamento
        pretrainer = PretrainerESMC(**pretrainer_config)
        pretrained_backbone_path = pretrainer.run()
        
        print(f"\n✅ Pré-treinamento concluído!")
        print(f"🎯 Backbone pré-treinado salvo em: {pretrained_backbone_path}")
        
        if not args.resume_pretraining:
            print(f"💡 Usando este backbone para fine-tuning...")
        
    elif (args.pretrain or args.resume_pretraining) and args.model.startswith("esm2"):
        print(f"⚠️  Pré-treinamento MLM ainda não suportado para ESM2")
        print(f"💡 Use modelos ESM-C (esmc_300m ou esmc_600m) para pré-treinamento")
    
    # Adicionar caminho do backbone pré-treinado ao config
    if pretrained_backbone_path:
        config["pretrained_backbone_path"] = pretrained_backbone_path
    
    # ────── Print configuration for confirmation ─────────────────────
    print(f"\n🧬 CONFIGURAÇÃO COMPLETA:")
    print(f"{'='*60}")
    print(f"📁 Dataset & Modelo:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Modelo: {args.model}")  
    print(f"   Tipo de vírus: {args.virus_type}")
    print(f"   Modo: {'Avaliação' if args.eval else 'Treinamento'}")
    
    print(f"\n⚙️  Hiperparâmetros de Treinamento:")
    print(f"   Épocas: {args.epochs}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Weight Decay: {args.weight_decay}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Max Length: {config['max_length']}")
    
    print(f"\n🧠 Configuração do Modelo:")
    print(f"   Dropout: {config.get('dropout', 'auto')}")
    print(f"   Freeze Backbone: {args.freeze_backbone}")
    
    print(f"\n⚖️  Pesos da Loss Function:")
    print(f"   Pos Class Weight: {args.pos_class_weight}")
    print(f"   Loss Weight Multiplier: {args.loss_weight_multiplier}")
    print(f"   💡 Efeito: {'Melhor precision' if args.pos_class_weight > 1.0 else 'Padrão' if args.pos_class_weight == 1.0 else 'Melhor recall'}")
    
    print(f"\n📊 Intervalos:")
    print(f"   Eval Interval: {args.eval_interval}")
    print(f"   Save Interval: {args.save_interval}")
    
    print(f"\n📈 Weights & Biases:")
    print(f"   Project: {config['project']}")
    print(f"   Entity: {config.get('entity', 'None')}")
    
    print(f"\n📂 Caminhos:")
    print(f"   Run name: {config['run_name']}")
    print(f"   Arquivos positivos: {config['train_pos'].split('/')[-1]}")
    print(f"   Arquivos negativos: {config['train_neg'].split('/')[-1]}")
    print(f"   Pasta de saída: {config['artifacts_path']}")
    print(f"{'='*60}")

    # Run trainer
    Trainer(**config).run()


if __name__ == "__main__":
    main()
