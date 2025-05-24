# optuna_eata_only.py --------------------------------------------------
import math, optuna, torch, os, random, numpy as np               # + random/np
from trainer import Trainer, EATAConfig
from config import get_config
# ── 1. semente global ────────────────────────────────────────────────
GLOBAL_SEED = 42

def seed_everything(seed: int = GLOBAL_SEED):
    """Garante reprodutibilidade em Python, NumPy e PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)            # idem para múltiplas GPUs
    torch.backends.cudnn.deterministic = True   # ↓‒ pode deixar mais lento!
    torch.backends.cudnn.benchmark = False

seed_everything()                               # chama logo na inicialização


# ── caminhos fixos ───────────────────────────────────────────────────
cfg = get_config()
ARTIFACTS = cfg["artifacts_path"]            # .../artefatos
RUN_NAME  = "run_20250428_210527"            # dir que já tem o .pt
CKPT_STEP = 14700                            # nome do peso
TRAIN_POS = cfg["train_pos"];  TRAIN_NEG = cfg["train_neg"]
VAL_POS   = cfg["val_pos"];    VAL_NEG   = cfg["val_neg"]
TEST_POS  = cfg["test_pos"];   TEST_NEG  = cfg["test_neg"]

os.environ["WANDB_MODE"] = "disabled"        # silencia W&B

# ── função-objetivo ──────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    hp = EATAConfig(
        lr        = trial.suggest_float("lr", 1e-6, 5e-2, log=True),
        momentum  = trial.suggest_float("momentum", 0.3, 0.99),
        steps     = trial.suggest_int("steps", 1, 3),
        d_margin  = trial.suggest_float("d_margin", 0.01, 1.0),
        e_margin  = trial.suggest_float("e_margin", 0.01, 1.2)*math.log(2),
        beta      = trial.suggest_float("beta", 1e-5, 1e-2, log=True),
        ema_alpha = trial.suggest_float("ema_alpha", 0.01, 0.99),
    )

    trainer = Trainer(
        train_pos=TRAIN_POS, train_neg=TRAIN_NEG,   # só p/ satisfazer paths
        val_pos=VAL_POS,   val_neg=VAL_NEG,
        test_pos=TEST_POS, test_neg=TEST_NEG,
        artifacts_path=ARTIFACTS,
        run_name=RUN_NAME,      # ← aponta p/ a pasta que já contém o .pt
        eval=True,              # modo avaliação
        step=CKPT_STEP,         # qual checkpoint usar
        eata_cfg=hp,
    )

    trainer.load_models()                       # carrega pesos
    f1 = trainer._compute_metrics_with_tta(trainer.test_loader,fixed_thr = 0.4)["f1"]
    return 1.0 - f1                            # minimizar ⇒ maximizar F1

# ── busca ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=50, timeout=14*3600)

    print("Melhor F1 :", 1.0 - study.best_value)
    print("HP ótimos :", study.best_params)
