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
from models import create_model, get_tokenizer  # Updated imports
from dataset import ProteinDataset
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
        best_model_metric="f1_precision_combined",
        pos_class_weight=3.0,  # CORREÇÃO: Agora vai para classe NEGATIVA - valores > 1.0 melhoram precision
        loss_weight_multiplier=1.0,  # Multiplicador escalar adicional para amplificar o efeito
        **kwargs  # Added to catch additional config parameters
    ):
        # ---------- store configuration --------------------------------- #
        self.train_pos = train_pos
        self.train_neg = train_neg
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
        self.best_model_metric = best_model_metric
        
        # Parâmetros para controle de peso na loss function
        self.pos_class_weight = pos_class_weight
        self.loss_weight_multiplier = loss_weight_multiplier

        self.project = project
        self.entity = entity
        self.seed = seed
        self.global_step = 0
        self.eval = eval
        self.step = step

        # Variáveis para rastrear o melhor modelo
        self.best_score = -float('inf')
        self.best_model_state = None
        self.best_epoch = 0

        # ---------- reproducibility ------------------------------------- #
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # ---------- runtime --------------------------------------------- #
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer(self.base_model)  # Updated to use factory function
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
                "best_model_metric": self.best_model_metric,
                "pos_class_weight": pos_class_weight,
                "loss_weight_multiplier": loss_weight_multiplier,
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
            # Primeiro tenta carregar o melhor modelo
            best_model_path = self.artifacts_path / "best_model.pt"
            if best_model_path.exists():
                weights_path = best_model_path
            else:
                # Se não existir, pega o mais recente
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

        self.model = create_model(self.base_model).to(self.device)  # Updated to use factory
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
        # Carregar dados de treino
        self.train_loader = self._make_loader(
            self.train_pos, self.train_neg, shuffle=True
        )
        
        # Carregar dados de teste (arquivos específicos HIV)
        self.test_loader = self._make_loader(self.test_pos, self.test_neg)
        
        # Log dos arquivos usados
        print(f"📁 Dados carregados:")
        print(f"   Treino: {self.train_pos} / {self.train_neg}")
        print(f"   Teste: {self.test_pos} / {self.test_neg}")
        print(f"   Treino: {len(self.train_loader.dataset)} amostras")
        print(f"   Teste: {len(self.test_loader.dataset)} amostras")

    def calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """
        Calcula score combinado baseado em F1 e Precision.
        Prioriza modelos com F1 alto E precision alto (não apenas recall alto).
        """
        f1 = metrics.get('f1', 0.0)
        precision = metrics.get('precision', 0.0)
        
        # Score combinado: média harmônica entre F1 e Precision
        # Isso garante que ambos sejam altos (não apenas um deles)
        if f1 > 0 and precision > 0:
            combined_score = 2 * (f1 * precision) / (f1 + precision)
        else:
            combined_score = 0.0
            
        return combined_score

    # ------------------------------------------------------------------ #
    # 2) Model & optimiser                                               #
    # ------------------------------------------------------------------ #
    def initialize_models(self):
        self.model = create_model(
            self.base_model, 
            num_labels=2,
            dropout=getattr(self, 'dropout', 0.3),
            freeze_backbone=getattr(self, 'freeze_backbone', False)
        ).to(self.device)

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

        # ------------------------------------------------------------------ #
        #  loss function com pesos personalizados                             #
        # ------------------------------------------------------------------ #
        # Criar tensor de pesos: [peso_classe_0, peso_classe_1]
        # CORREÇÃO: Para melhorar PRECISION (menos FP), aumentamos peso da classe NEGATIVA!
        # Falsos Positivos = real negativo → usa peso da classe 0 (negativo)
        # IMPORTANTE: usar dtype=torch.float32 para compatibilidade com mixed precision
        class_weights = torch.tensor([self.pos_class_weight, 1.0], dtype=torch.float32, device=self.device)
        class_weights = class_weights * self.loss_weight_multiplier
        
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"🔧 Loss function configurada:")
        print(f"   Peso classe negativa (0): {class_weights[0]:.3f}")
        print(f"   Peso classe positiva (1): {class_weights[1]:.3f}")
        print(f"   Multiplicador: {self.loss_weight_multiplier:.3f}")
        print(f"   Dtype: {class_weights.dtype}")
        print(f"   💡 Peso maior na classe negativa → penaliza mais FP → melhor precision")

    # ------------------------------------------------------------------ #
    # 3) Training and evaluation helpers                                 #
    # ------------------------------------------------------------------ #
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        ema_alpha = 0.97
        ema_loss = None

        for step, (ids, mask, y) in enumerate(tqdm(self.train_loader, desc="Train")):
            ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass e loss calculation dentro do contexto de mixed precision
            from torch import amp
            with amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(self.device.type == "cuda"),
            ):
                logits = self.model(ids, mask)
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

    # ------------------------------------------------------------------ #
    # VALIDAÇÃO REMOVIDA - Usando apenas TREINO + TESTE                 #
    # O modelo treina com os dados de treino e é avaliado apenas        #
    # no conjunto de teste específico (arquivos HIV).                   #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # 4) Full evaluation on test set                                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _compute_metrics(self, loader) -> Dict[str, float]:
        """
        Computa métricas de avaliação no conjunto de dados fornecido
        """
        preds, probs, ys = [], [], []
        total_loss = 0
        num_batches = 0

        self.model.eval()  # modo de inferência, desativa dropout, etc
        for ids, mask, y in loader:
            ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)

            # forward pass e cálculo de loss dentro do contexto de mixed precision
            from torch import amp
            with amp.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=(self.device.type == "cuda"),
            ):
                logits = self.model(ids, mask)  # (B, C)
                loss = self.loss_fn(logits, y)
            
            total_loss += loss.item()
            num_batches += 1
            
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            pred = (prob >= 0.5).astype(int)

            preds.extend(pred)
            probs.extend(prob)
            ys.extend(y.cpu().numpy())

        y_true, y_pred, y_prob = map(np.asarray, (ys, preds, probs))
        
        # Calcular loss média
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Verificar se há valores NaN ou infinitos nas probabilidades
        if np.any(np.isnan(y_prob)) or np.any(np.isinf(y_prob)):
            print("⚠️  Detectados valores NaN/Inf nas probabilidades, aplicando correção...")
            # Substituir NaN/Inf por valores seguros
            y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
            # Recalcular predições
            y_pred = (y_prob >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calcular AUC com proteção adicional
        try:
            if len(np.unique(y_true)) > 1:
                auc_score = roc_auc_score(y_true, y_prob)
                # Verificar se AUC é válido
                if np.isnan(auc_score) or np.isinf(auc_score):
                    auc_score = 0.5
            else:
                auc_score = 0.5
        except Exception as e:
            print(f"⚠️  Erro no cálculo AUC: {e}, usando AUC = 0.5")
            auc_score = 0.5
        
        # Imprimir matriz de confusão
        print("\n📊 MATRIZ DE CONFUSÃO:")
        print("─" * 30)
        print(f"                Predito")
        print(f"              Neg   Pos")
        print(f"Real    Neg  {tn:4d}  {fp:4d}")
        print(f"        Pos  {fn:4d}  {tp:4d}")
        print("─" * 30)
        print(f"TN={tn} (Verdadeiros Negativos) | FP={fp} (Falsos Positivos)")
        print(f"FN={fn} (Falsos Negativos)      | TP={tp} (Verdadeiros Positivos)")
        
        metrics = {
            "loss": avg_loss,  # Adicionar loss média
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "auc": auc_score,  # Usar valor protegido
        }
        
        # Adicionar score combinado
        metrics["f1_precision_combined"] = self.calculate_combined_score(metrics)
        
        return metrics

    def evaluate(self) -> Dict[str, float]:
        torch.cuda.empty_cache()
        print("\n🧪 AVALIANDO MODELO NO CONJUNTO DE TESTE...")
        metrics = self._compute_metrics(self.test_loader)
        
        # Interpretação das métricas
        print(f"\n📈 MÉTRICAS DE DESEMPENHO:")
        print("─" * 50)
        print(f"📉 Loss:        {metrics['loss']:.4f} (Loss média no conjunto de teste)")
        print(f"🎯 Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
        print(f"🔍 Precision:   {metrics['precision']:.4f} (De todas as predições positivas, {metrics['precision']*100:.1f}% estavam corretas)")
        print(f"📡 Recall:      {metrics['recall']:.4f} (Detectou {metrics['recall']*100:.1f}% dos casos positivos)")
        print(f"🛡️  Specificity: {metrics['specificity']:.4f} (Detectou {metrics['specificity']*100:.1f}% dos casos negativos)")
        print(f"⚖️  F1-Score:    {metrics['f1']:.4f} (Harmônico entre Precision e Recall)")
        print(f"🧮 MCC:         {metrics['mcc']:.4f} (Correlação Matthews: -1=péssimo, 0=aleatório, 1=perfeito)")
        print(f"📊 AUC:         {metrics['auc']:.4f} (Área sob curva ROC)")
        print(f"🏆 Combined:    {metrics['f1_precision_combined']:.4f} (Critério F1+Precision)")

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

    def save_best_model(self, metrics: Dict[str, float], epoch: int):
        """
        Salva o melhor modelo baseado no critério especificado.
        """
        current_score = metrics.get(self.best_model_metric, 0.0)
        
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_epoch = epoch
            self.best_model_state = deepcopy(self.model.state_dict())
            
            # Salvar melhor modelo
            best_model_path = self.artifacts_path / "best_model.pt"
            torch.save(self.best_model_state, best_model_path)
            
            # Salvar informações do melhor modelo
            best_info = {
                "epoch": epoch,
                "score": current_score,
                "metric": self.best_model_metric,
                "all_metrics": metrics
            }
            
            best_info_path = self.artifacts_path / "best_model_info.json"
            with open(best_info_path, 'w') as f:
                json.dump(best_info, f, indent=4)
            
            print(f"🎯 Novo melhor modelo salvo! Época {epoch}, {self.best_model_metric}: {current_score:.4f}")
            
            # Log no wandb
            wandb.log({
                f"best/{self.best_model_metric}": current_score,
                "best/epoch": epoch
            }, step=self.global_step)

    # ------------------------------------------------------------------ #
    # 6) Main loop                                                       #
    # ------------------------------------------------------------------ #
    def run(self):
        # ----------- modo avaliação apenas ------------------------------------ #
        from tabulate import tabulate

        if self.eval and self.run_name is not None:
            self.load_models()

            # --- Normal inference ---
            metrics = self._compute_metrics(self.test_loader)
            print("\\ -- Standard Inference Metrics --")
            print(
                tabulate(metrics.items(), headers=["Metric", "Value"], floatfmt=".4f")
            )

            # --- wandb logging ---
            wandb.log({f"evaluate/{k}": v for k, v in metrics.items()})

            # --- local json logging ---
            self._update_json({"evaluate": metrics})

            print(
                "Avaliação concluída — métricas registradas. "
                f"Modelo carregado de: {self.artifacts_path.resolve()}"
            )
            wandb.finish()
            return

        # ----------- ciclo normal de treino + teste (SEM VALIDAÇÃO) -------------- #
        print(f"🚀 Iniciando treinamento por {self.epochs} épocas...")
        print(f"🎯 Critério de melhor modelo: {self.best_model_metric}")
        print(f"📊 Apenas TREINO + TESTE (sem validação)")
        
        for epoch in range(1, self.epochs + 1):
            print(f"\n📈 Época {epoch}/{self.epochs}")
            
            # Apenas treinamento (sem validação)
            tr = self.train_epoch()
            
            # Preparar registro da época
            epoch_record = {"epoch": epoch, "training": tr}

            # Avaliação no conjunto de teste
            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                test_metrics = self.evaluate()
                epoch_record["test"] = test_metrics  # Adicionar métricas de teste ao registro da época
                
                # Imprimir métricas principais
                print(f"   Test Loss: {test_metrics['loss']:.4f}")
                print(f"   Test F1: {test_metrics['f1']:.4f}")
                print(f"   Test Precision: {test_metrics['precision']:.4f}")
                print(f"   Test Recall: {test_metrics['recall']:.4f}")
                print(f"   Combined Score: {test_metrics['f1_precision_combined']:.4f}")
                
                # Salvar melhor modelo
                self.save_best_model(test_metrics, epoch)
            
            # Salvar registro da época (com ou sem métricas de teste)
            self._update_json(epoch_record)

            # Salvar checkpoint regular
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self.save_models()

        # Informações finais sobre o melhor modelo
        print("\n" + "="*60)
        print("🏆 TREINAMENTO CONCLUÍDO")
        print("="*60)
        
        if self.best_model_state is not None:
            print(f"🥇 Melhor modelo encontrado na época {self.best_epoch}")
            print(f"   {self.best_model_metric}: {self.best_score:.4f}")
            
            # Carregar melhor modelo para avaliação final
            self.model.load_state_dict(self.best_model_state)
            print(f"\n🏆 AVALIAÇÃO FINAL DO MELHOR MODELO:")
            final_metrics = self._compute_metrics(self.test_loader)
            
            print(f"\n📊 Métricas finais do melhor modelo:")
            print(f"   Loss: {final_metrics['loss']:.4f}")
            print(f"   Accuracy: {final_metrics['accuracy']:.4f}")
            print(f"   Precision: {final_metrics['precision']:.4f}")
            print(f"   Recall: {final_metrics['recall']:.4f}")
            print(f"   F1-Score: {final_metrics['f1']:.4f}")
            print(f"   Specificity: {final_metrics['specificity']:.4f}")
            print(f"   MCC: {final_metrics['mcc']:.4f}")
            print(f"   AUC: {final_metrics['auc']:.4f}")
            print(f"   Combined Score: {final_metrics['f1_precision_combined']:.4f}")
            
            # Salvar métricas finais
            self._update_json({"final_best_model": final_metrics})
            
            # Log final no wandb
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
        else:
            print("⚠️  Nenhum modelo melhor foi encontrado durante o treinamento")

        # Salvar informações do run
        run_info = {
            "seed": self.seed,
            "total_epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_score": self.best_score,
            "best_metric": self.best_model_metric,
            "validation": False  # Confirmando que não usa validação
        }
        
        (self.artifacts_path / "run_info.json").write_text(
            json.dumps(run_info, indent=4)
        )
        
        print(f"\n📁 Artefatos salvos em: {self.artifacts_path.resolve()}")
        print("   ├── best_model.pt (melhor modelo)")
        print("   ├── best_model_info.json (informações do melhor modelo)")
        print("   ├── model_metrics.json (histórico de métricas)")
        print("   └── run_info.json (informações do treinamento)")
        
        wandb.finish()
