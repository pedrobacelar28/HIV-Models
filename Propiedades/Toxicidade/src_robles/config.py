def get_config():
    config = {
        # ── Data paths ───────────────────────────────────────────── #
        "train_pos": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Treino/sim_train.txt",
        "train_neg": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Treino/nao_train.txt",
        "val_pos": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Treino/sim_val.txt",
        "val_neg": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Treino/nao_val.txt",
        "test_pos": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Teste/sim_independent.txt",
        "test_neg": "/scratch/pedro.bacelar/HIV-Models/Propiedades/Toxicidade/Datasets/Teste/nao_independent.txt",
        # ── Training hyperparameters ─────────────────────────────── #
        "lr": 5e-3,
        "weight_decay": 0.01,
        "batch_size": 8,
        "max_length": 1200,
        "epochs": 30,
        "eval_interval": 1,
        "save_interval": 1,
        "base_model": "esmc_300m",
        # ── Weights & Biases setup ──────────────────────────────── #
        "project": "protein-toxicity",
        "entity": None,  # or "your-wandb-entity"
        "run_name": None,
        # ── Reproducibility and output ───────────────────────────── #
        "seed": 42,
        "artifacts_path": "/scratch/pedroroblesduten/hiv_bracis/HIV-Models/Propiedades/Toxicidade/src_robles/artefatos",
        "eval": False,
        "step": None,
    }
    return config
