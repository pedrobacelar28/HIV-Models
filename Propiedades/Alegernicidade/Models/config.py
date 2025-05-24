def get_config():
    config = {
        # ── Data paths ───────────────────────────────────────────── #
        "train_pos": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Treino/sim_PSIBiology_train.txt",
        "train_neg": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Treino/nao_PSIBiology_train.txt",
        "val_pos": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Treino/sim_PSIBiology_val.txt",
        "val_neg": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Treino/nao_PSIBiology_val.txt",
        "test_pos": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Testes/sim_NESG.txt",
        "test_neg": "/scratch/pedrobacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Testes/nao_NESG.txt",
        # ── Training hyperparameters ─────────────────────────────── #
        "lr": 5e-7,
        "weight_decay": 0.01,
        "batch_size": 4,
        "max_length": 1200,
        "epochs": 10,
        "eval_interval": 1,
        "save_interval": 1,
        "base_model": "esmc_600m",
        # ── Weights & Biases setup ──────────────────────────────── #
        "project": "protein-solubility",
        "entity": None,  # or "your-wandb-entity"
        "run_name": None,
        # ── Reproducibility and output ───────────────────────────── #
        "seed": 42,
        "artifacts_path": "/scratch/pedrobacelar/HIV-Models/src_robles/artefatos",
        "eval": False,
        "step": 2500,
    }
    return config
