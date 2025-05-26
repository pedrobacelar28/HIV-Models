import os
from pathlib import Path
import datetime

def get_config(dataset_type="B", model_type="esmc_300m", virus_type="Tudo"):
    """
    Configuração de caminhos e configurações básicas para treinar modelos em diferentes datasets.
    
    Args:
        dataset_type: str - "B", "MHC1" ou "MHC2"
        model_type: str - "esmc_300m", "esmc_600m", "esm2_t33_650M_UR50D", etc.
        virus_type: str - "Base", "Tudo", "Virus", "Lent", "Retro" (especifica qual arquivo usar)
    """
    
    # Caminho base do projeto
    base_path = Path(__file__).parent
    
    # Definir caminhos dos dados baseados no dataset_type
    dataset_configs = {
        "B": {
            "positive_prefix": "simB",
            "negative_prefix": "naoB",
            "test_positive": "simBHIV.txt",
            "test_negative": "naoBHIV.txt",
            "test_folder": "Bcell"
        },
        "MHC1": {
            "positive_prefix": "simMHC1",
            "negative_prefix": "naoMHC1", 
            "test_positive": "simMHC1HIV.txt",
            "test_negative": "naoMHC1HIV.txt",
            "test_folder": "MHC1"
        },
        "MHC2": {
            "positive_prefix": "simMHC2", 
            "negative_prefix": "naoMHC2",
            "test_positive": "simMHC2HIV.txt", 
            "test_negative": "naoMHC2HIV.txt",
            "test_folder": "MHC2"
        }
    }
    
    if dataset_type not in dataset_configs:
        raise ValueError(f"dataset_type deve ser um de: {list(dataset_configs.keys())}")
    
    dataset_config = dataset_configs[dataset_type]
    dataset_path = base_path / dataset_type
    
    # Construir nomes dos arquivos de treinamento baseados no virus_type
    if virus_type == "Base":
        # Arquivos base sem sufixo (ex: simB.txt, naoB.txt)
        pos_filename = f"{dataset_config['positive_prefix']}.txt"
        neg_filename = f"{dataset_config['negative_prefix']}.txt"
    else:
        # Arquivos com sufixo (ex: simBTudo.txt, naoBTudo.txt)
        pos_filename = f"{dataset_config['positive_prefix']}{virus_type}.txt"
        neg_filename = f"{dataset_config['negative_prefix']}{virus_type}.txt"
    
    # Verificar se os arquivos de treinamento existem
    pos_file = dataset_path / pos_filename
    neg_file = dataset_path / neg_filename
    
    if not pos_file.exists():
        raise FileNotFoundError(f"Arquivo de treinamento não encontrado: {pos_file}")
    if not neg_file.exists():
        raise FileNotFoundError(f"Arquivo de treinamento não encontrado: {neg_file}")
    
    # Caminhos dos arquivos de teste (Inferencia_HIV está no mesmo nível que Modelos)
    test_base_path = base_path.parent / "Inferencia_HIV" / dataset_config["test_folder"]
    test_pos_file = test_base_path / dataset_config["test_positive"]
    test_neg_file = test_base_path / dataset_config["test_negative"]
    
    # Verificar se os arquivos de teste existem
    if not test_pos_file.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_pos_file}")
    if not test_neg_file.exists():
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {test_neg_file}")
    
    # Configurações padrão de modelos (usadas apenas se não especificado no main.py)
    model_defaults = {
        "esmc_300m": {
            "base_model": "esmc_300m",
            "max_length": 60,
            "batch_size": 8,
            "dropout": 0.3,
            "freeze_backbone": False
        },
        "esmc_600m": {
            "base_model": "esmc_600m", 
            "max_length": 60,
            "batch_size": 4,
            "dropout": 0.3,
            "freeze_backbone": False
        },
        "esm2_t33_650M_UR50D": {
            "base_model": "esm2_t33_650M_UR50D",
            "max_length": 60,
            "batch_size": 6,
            "dropout": 0.3,
            "freeze_backbone": False
        },
        "esm2_t36_3B_UR50D": {
            "base_model": "esm2_t36_3B_UR50D",
            "max_length": 60,
            "batch_size": 2,
            "dropout": 0.3,
            "freeze_backbone": False
        }
    }
    
    if model_type not in model_defaults:
        raise ValueError(f"model_type deve ser um de: {list(model_defaults.keys())}")
    
    model_config = model_defaults[model_type]
    
    # Criar run_name único com timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_type}_{virus_type}_{model_type}_{timestamp}"
    
    config = {
        # ── Data paths ───────────────────────────────────────────── #
        "train_pos": str(pos_file),
        "train_neg": str(neg_file),
        "test_pos": str(test_pos_file),   # Arquivos específicos de teste HIV
        "test_neg": str(test_neg_file),   # Arquivos específicos de teste HIV
        
        # ── Configurações padrão do modelo (podem ser sobrescritas no main.py) ─ #
        "base_model": model_config["base_model"],
        "max_length": model_config["max_length"],
        "batch_size": model_config["batch_size"],
        "dropout": model_config["dropout"],
        "freeze_backbone": model_config["freeze_backbone"],
        
        # ── Weights & Biases setup ──────────────────────────────── #
        "project": f"protein-{dataset_type.lower()}",
        "entity": None,
        "run_name": run_name,
        
        # ── Output and metadata ──────────────────────────────────── #
        "artifacts_path": str(base_path / dataset_type / "model"),
        "dataset_type": dataset_type,
        "model_type": model_type,
        "virus_type": virus_type,
        
        # ── Critério de melhor modelo ──────────────────────────── #
        "best_model_metric": "f1_precision_combined",
    }
    
    return config
