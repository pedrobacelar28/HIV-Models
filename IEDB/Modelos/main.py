import argparse
import os
from pathlib import Path
from config import get_config
from trainer import Trainer

# ================================================================== #
# 🔧 CONFIGURAÇÃO DE DEFAULTS (EDITE AQUI PARA FACILITAR O USO)    #
# ================================================================== #
# Edite estes valores para evitar usar argumentos na linha de comando

DEFAULT_DATASET = "B"                    # Opções: "B", "MHC1", "MHC2"
DEFAULT_MODEL = "esm2_t33_650M_UR50D"              # Opções: "esmc_300m", "esmc_600m", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"
DEFAULT_VIRUS_TYPE = "Base"              # Opções: "Base", "Tudo", "Virus", "Lent", "Retro"
DEFAULT_EPOCHS = 5                       # Número de épocas (ajuste conforme necessário)
DEFAULT_EVAL_MODE = False                # True = apenas avaliação, False = treinamento

# ---- Para modo AVALIAÇÃO: especifique a run que quer testar ---- #
DEFAULT_RUN_NAME = None                  # Ex: "B_Base_esmc_300m_20241218_143022" ou None para usar o mais recente

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
    
    # Modo de execução
    parser.add_argument(
        "--eval", action="store_true", default=DEFAULT_EVAL_MODE,
        help="Run in evaluation mode (default: False)"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, choices=["B", "MHC1", "MHC2"],
        help=f"Dataset to use: B, MHC1, or MHC2 (default: {DEFAULT_DATASET})"
    )
    
    # Model selection
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=["esmc_300m", "esmc_600m", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D"],
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    
    # Virus type selection
    parser.add_argument(
        "--virus-type", type=str, default=DEFAULT_VIRUS_TYPE, 
        choices=["Base", "Tudo", "Virus", "Lent", "Retro"],
        help=f"Type of virus data to use (default: {DEFAULT_VIRUS_TYPE}). 'Base' uses files without suffix (e.g., simB.txt)"
    )
    
    # Epochs override
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})"
    )
    
    # Step for evaluation
    parser.add_argument(
        "--step", type=int, default=None,
        help="Specific step to load for evaluation (default: melhor modelo)"
    )
    
    # Run específico para avaliação
    parser.add_argument(
        "--run-name", type=str, default=DEFAULT_RUN_NAME,
        help=f"Specific run name to evaluate (default: {DEFAULT_RUN_NAME or 'most recent'}). Use --list-runs to see available runs"
    )
    
    # Listar runs disponíveis
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List available runs for the specified dataset/model/virus-type and exit"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Se pediu para listar runs, faz isso e sai
    if args.list_runs:
        list_available_runs(args.dataset, args.model, args.virus_type)
        return
    
    # Get configuration with specified parameters
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
    # Se não está em modo avaliação, usa o run_name gerado automaticamente

    # Update config with command line arguments
    config["eval"] = args.eval
    config["step"] = args.step
    config["epochs"] = args.epochs  # Override epochs if specified
    
    # Print configuration for confirmation
    print(f"\n🧬 Configuração:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Modelo: {args.model}")
    print(f"   Tipo de vírus: {args.virus_type}")
    print(f"   Épocas: {args.epochs}")
    print(f"   Modo: {'Avaliação' if args.eval else 'Treinamento'}")
    print(f"   Run name: {config['run_name']}")
    print(f"   Arquivos positivos: {config['train_pos'].split('/')[-1]}")
    print(f"   Arquivos negativos: {config['train_neg'].split('/')[-1]}")
    print(f"   Pasta de saída: {config['artifacts_path']}")
    print("=" * 60)

    # Run trainer
    Trainer(**config).run()


if __name__ == "__main__":
    main()
