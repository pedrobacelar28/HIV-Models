# trainer_esmc_wandb.py
# -*- coding: utf-8 -*-
"""
ESMC fine-tuning for binary protein-sequence classification
Object-oriented refactor + Weights & Biases (wandb) logging
"""

from __future__ import annotations
import json, random, datetime, os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

import wandb
from esm.tokenization import get_esmc_model_tokenizers
from models import ESMCForSequenceClassification  # ⚠️ must exist
from dataset import ProteinDataset
from tta.eata import EATA
from copy import deepcopy

# top of trainer_esmc_wandb.py
from transformers.optimization import get_cosine_schedule_with_warmup  # NEW


# ---------------------------------------------------------------------------- #
#  Trainer                                                                     #
# ---------------------------------------------------------------------------- #
class Trainer:
    # ---------------------------------------------------------------------- #
    def __init__(
        self,
        train_pos,
        train_neg,
        val_pos,
        val_neg,
        test_pos,
        test_neg,
        artifacts_path,
        lr=1e-5,
        weight_decay=0.01,
        batch_size=8,
        max_length=1200,
        epochs=2,
        eval_interval=1,
        save_interval=1,
        base_model="esmc_300m",
        project="protein-toxicity",
        entity=None,
        run_name=None,
        seed=42,
        eval=False,
        step=None,
    ):
        # ---------- store configuration --------------------------------- #
        self.train_pos = train_pos
        self.train_neg = train_neg
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.test_pos = test_pos
        self.test_neg = test_neg

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.base_model = base_model

        self.project = project
        self.entity = entity
        self.seed = seed
        self.global_step = 0
        self.eval = eval
        self.step = step

        # ---------- reproducibility ------------------------------------- #
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ---------- runtime --------------------------------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_esmc_model_tokenizers()
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name if run_name is not None else f"run_{time_stamp}"
        # ---------- wandb init ------------------------------------------ #
        wandb.init(
            project=project,
            entity=entity,
            name=self.run_name,
            config={  # explicit config dictionary for wandb logging
                "train_pos": self.train_pos,
                "train_neg": self.train_neg,
                "val_pos": self.val_pos,
                "val_neg": self.val_neg,
                "test_pos": self.test_pos,
                "test_neg": self.test_neg,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "epochs": self.epochs,
                "eval_interval": self.eval_interval,
                "save_interval": self.save_interval,
                "base_model": self.base_model,
                "seed": self.seed,
            },
            reinit="finish_previous",
        )
        self.artifacts_path = Path(artifacts_path) / self.run_name
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.artifacts_path / "model_metrics.json"

        # ---------- build pipeline -------------------------------------- #
        self.load_data()
        self.initialize_models()

    def log_gradient_stats(self):
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return

        grad_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
        )
        grad_max = max(g.max().item() for g in grads)
        grad_min = min(g.min().item() for g in grads)

        wandb.log(
            {
                "grad/norm": grad_norm.item(),
                "grad/max": grad_max,
                "grad/min": grad_min,
            },
            step=self.global_step,
        )

    def load_models(self):
        if hasattr(self, "step") and self.step is not None:
            weights_path = self.artifacts_path / f"pytorch_model_step{self.step}.pt"
        else:
            candidates = list(self.artifacts_path.glob("pytorch_model_step*.pt"))
            if not candidates:
                raise FileNotFoundError(
                    "Nenhum checkpoint encontrado na pasta de artefatos."
                )

            def extract_step(path):
                try:
                    return int(path.stem.split("step")[-1])
                except ValueError:
                    return -1

            weights_path = max(candidates, key=extract_step)

        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {weights_path}")

        self.model = ESMCForSequenceClassification(base_model=self.base_model).to(
            self.device
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        print(f"✅ Modelo carregado de: {weights_path.name}")

    # ------------------------------------------------------------------ #
    # 1) Data loading                                                    #
    # ------------------------------------------------------------------ #
    def _load_sequences(self, path: str, label: int) -> Tuple[list, list]:
        seqs, labs = [], []
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s:
                    seqs.append(s)
                    labs.append(label)
        return seqs, labs

    def _make_loader(self, pos, neg, shuffle=False):
        p_seq, p_lab = self._load_sequences(pos, 1)
        n_seq, n_lab = self._load_sequences(neg, 0)
        ds = ProteinDataset(
            p_seq + n_seq, p_lab + n_lab, self.tokenizer, self.max_length
        )
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def load_data(self):
        self.train_loader = self._make_loader(
            self.train_pos, self.train_neg, shuffle=True
        )
        self.val_loader = self._make_loader(self.val_pos, self.val_neg)
        self.test_loader = self._make_loader(self.test_pos, self.test_neg)

    # ------------------------------------------------------------------ #
    # 2) Model & optimiser                                               #
    # ------------------------------------------------------------------ #
    def initialize_models(self):
        self.model = ESMCForSequenceClassification(base_model=self.base_model).to(
            self.device
        )

        # ------------------------------------------------------------------ #
        #  optimizer                                                         #
        # ------------------------------------------------------------------ #
        # discriminative LRs (backbone vs head) are optional – keep simple here
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ------------------------------------------------------------------ #
        #  scheduler: warm-up 5 %  →  cosine to 0                             #
        # ------------------------------------------------------------------ #
        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = int(0.05 * total_steps)  # 5 % warm-up

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # 3) Epoch helpers                                                   #
    # ------------------------------------------------------------------ #
    def _forward(self, ids, mask):
        """Forward pass with optional loss (mixed precision on CUDA)."""
        from torch import amp

        with amp.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=(self.device.type == "cuda"),
        ):
            logits = self.model(ids, mask)
        return logits

    # ------------------------ training -------------------------------- #
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        ema_alpha = 0.97
        ema_loss = None

        for step, (ids, mask, y) in enumerate(tqdm(self.train_loader, desc="Train")):
            ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self._forward(ids, mask)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()
            lr_now = self.scheduler.get_last_lr()[0]  # list → scalar

            loss_val = loss.item()
            total_loss += loss_val

            # === EMA calculation ===
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

            # === Logging ===
            self.global_step += 1
            wandb.log(
                {
                    "train/loss": loss_val,  # batch loss (low-opacity)
                    "train/loss_ema": ema_loss,  # smoothed loss (bold line)
                    "train/lr": lr_now,
                },
                step=self.global_step,
            )

            self.log_gradient_stats()

        return {"loss": total_loss / len(self.train_loader)}

    # ------------------------ validation ------------------------------ #
    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        ema_alpha = 0.97
        ema_loss = None

        for ids, mask, y in tqdm(self.val_loader, desc="Val"):
            ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
            with torch.no_grad():
                logits = self._forward(ids, mask)
                loss = self.loss_fn(logits, y)

            loss_val = loss.item()
            total_loss += loss_val

            # === EMA calculation ===
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * loss_val

            # === Logging ===
            wandb.log(
                {
                    "val/loss": loss_val,  # raw batch val loss
                    "val/loss_ema": ema_loss,  # smoothed val loss
                },
                step=self.global_step,
            )

        return {"loss": total_loss / len(self.val_loader)}

    # ------------------------------------------------------------------ #
    # 4) Full evaluation on test set                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_metrics(self, loader) -> Dict[str, float]:
        self.model.eval()
        ys, preds, probs = [], [], []
        for ids, mask, y in tqdm(loader, desc="Test"):
            ids, mask = ids.to(self.device), mask.to(self.device)
            with torch.no_grad():
                logits = self._forward(ids, mask)
            prob = F.softmax(logits, 1)[:, 1]
            ys.extend(y.numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(prob.cpu().numpy())

        y_true, y_pred, y_prob = map(np.asarray, (ys, preds, probs))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        }

        # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 4-b) Métricas com Test-Time Adaptation (EATA + Fisher)              #
    # ------------------------------------------------------------------ #
    def _estimate_fisher(
        self,
        id_loader: DataLoader,
        n_batches: int = 16,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Calcula a diagonal da matriz de Fisher (ω) para os parâmetros
        adaptáveis (γ/β das normalizações) usando rótulos-pseudo.

        • Passa `n_batches` mini-batches do conjunto ID (val_loader)
        • Acumula ||∇θ Lce||² por parâmetro  ➜  ω = média
        • Retorna {nome: (ω, θ₀)}  onde θ₀ = valor original (cópia)
        """
        self.model.eval()
        adaptable_names, _ = EATA._collect_adaptable_params(self.model)
        # buffers de acumulação
        fisher_diag = {
            n: torch.zeros_like(p, device=self.device)
            for n, p in self.model.named_parameters()
            if n in adaptable_names
        }

        seen = 0
        for i, (ids, mask, _) in enumerate(id_loader):
            if i >= n_batches:  # evita custo excessivo
                break
            ids, mask = ids.to(self.device), mask.to(self.device)

            # --- forward + rótulos-pseudo ---
            logits = self._forward(ids, mask)  # (B, C)
            pseudo = logits.argmax(1)
            loss = F.cross_entropy(logits, pseudo, reduction="sum")

            # --- back-prop por lote inteiro ---
            self.model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n in fisher_diag and p.grad is not None:
                        fisher_diag[n] += p.grad.detach() ** 2
            seen += ids.size(0)

        # média e empacotamento (ω, θ₀)
        for n in fisher_diag:
            fisher_diag[n] = (
                fisher_diag[n] / float(seen),
                self.model.state_dict()[n].detach().clone(),
            )
        return fisher_diag

    def _compute_metrics_with_tta(self, loader) -> Dict[str, float]:
        """
        Aplica EATA com regularização Fisher no conjunto de teste
        e devolve métricas finais (sem usar rótulos durante a adaptação).
        """
        # 1) ------------------------------------------------------------- #
        #    Estima Fisher numa pequena porção ID (validation).
        #    16 mini-batches de 8 seq.  ➜  ≈ 128 amostras são suficientes.
        fishers = self._estimate_fisher(self.val_loader, n_batches=16)

        # 2) ------------------------------------------------------------- #
        #    Instancia o motor EATA já com o Fisher
        eata_engine = EATA(
            model=deepcopy(self.model),
            lr=2e-4,  # LR menor + Fisher = mais estável
            steps=3,  # 3 updates por batch melhora recall
            episodic=False,  # reinicia estado a cada batch
            d_margin=0.05,  # filtra amostras redundantes
            fishers=0.0,
            device=self.device,
        )

        # 3) ------------------------------------------------------------- #
        ys, preds, probs = eata_engine.run_eata(loader)

        # 4) ------------------------------------------------------------- #
        y_true = np.asarray(ys)
        y_pred = np.asarray(preds)
        y_prob = np.asarray(probs)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        }

    def evaluate(self) -> Dict[str, float]:
        torch.cuda.empty_cache()
        metrics = self._compute_metrics(self.test_loader)

        wandb.log({f"test/{k}": v for k, v in metrics.items()})
        return metrics

    # ------------------------------------------------------------------ #
    # 5) Saving                                                          #
    # ------------------------------------------------------------------ #
    def _update_json(self, record: dict):
        data = {}
        if self.metrics_file.exists():
            data = json.loads(self.metrics_file.read_text())

        if "epoch" in record:
            data.setdefault("epochs", []).append(record)
        else:
            # Permite salvar "test", "evaluate" ou outros registros únicos
            data.update(record)

        self.metrics_file.write_text(json.dumps(data, indent=4))

    def save_models(self):
        """
        Locally save model weights under self.artifacts_path, with step in filename.
        """
        # — model weights —
        weights_path = self.artifacts_path / f"pytorch_model_step{self.global_step}.pt"
        torch.save(self.model.state_dict(), weights_path)

    # ------------------------------------------------------------------ #
    # 6) Main loop                                                       #
    # ------------------------------------------------------------------ #
    def run(self):
        # ----------- modo avaliação apenas ------------------------------------ #
        from tabulate import tabulate

        if self.eval and self.run_name is not None:
            self.load_models()

            # --- TTA ---
            tta_metrics = self._compute_metrics_with_tta(self.test_loader)
            print("\n-- Test-Time Adaptation (EATA) Metrics --")
            print(
                tabulate(
                    tta_metrics.items(), headers=["Metric", "Value"], floatfmt=".4f"
                )
            )

            # --- Normal inference ---
            metrics = self._compute_metrics(self.test_loader)
            print("\\ -- Standard Inference Metrics --")
            print(
                tabulate(metrics.items(), headers=["Metric", "Value"], floatfmt=".4f")
            )

            # --- wandb logging ---
            wandb.log({f"evaluate/{k}": v for k, v in metrics.items()})
            wandb.log({f"evaluate_tta/{k}": v for k, v in tta_metrics.items()})

            # --- local json logging ---
            self._update_json({"evaluate": metrics})
            self._update_json({"evaluate_tta": tta_metrics})

            print(
                "Avaliação concluída — métricas registradas. "
                f"Modelo carregado de: {self.artifacts_path.resolve()}"
            )
            wandb.finish()
            return

        # ----------- ciclo normal de treino + validação + teste --------------- #
        for epoch in range(1, self.epochs + 1):
            tr = self.train_epoch()
            val = self.val_epoch()
            self._update_json({"epoch": epoch, "training": tr, "validation": val})

            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                test_metrics = self.evaluate()
                self._update_json({"test": test_metrics})

            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self.save_models()

        (self.artifacts_path / "run_info.json").write_text(
            json.dumps({"seed": torch.initial_seed()}, indent=4)
        )
        print("Finished — artifacts em:", self.artifacts_path.resolve())
        wandb.finish()
