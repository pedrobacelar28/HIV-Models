# -*- coding: utf-8 -*-
"""
Finetuning ESMC para classificação binária de sequências,
com attention pooling, threshold calibrado, early stopping, LR scheduler e k-fold CV
"""

from __future__ import annotations

import os
import json
import random
import datetime
import time
from pathlib import Path
from types import SimpleNamespace
from torch.amp import autocast, GradScaler
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)
from tqdm.auto import tqdm

# ESM‑C imports
from esm.pretrained import load_local_model
from esm.tokenization import get_esmc_model_tokenizers

# Configuração de seed para reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

tokenizer = get_esmc_model_tokenizers()

class ESMCForSequenceClassification(nn.Module):
    def __init__(self, num_labels: int = 2, base_model_name: str = "esmc_600m"):
        super().__init__()
        self.esmc = load_local_model(base_model_name)
        
        if hasattr(self.esmc, 'embed') and hasattr(self.esmc.embed, 'embedding_dim'):
             self.hidden_size = self.esmc.embed.embedding_dim
        elif hasattr(self.esmc, 'config') and hasattr(self.esmc.config, 'hidden_size'):
             self.hidden_size = self.esmc.config.hidden_size
        elif hasattr(self.esmc, 'args') and hasattr(self.esmc.args, 'embed_dim'):
             self.hidden_size = self.esmc.args.embed_dim
        else:
            try:
                dummy_input = torch.ones(1, 10, dtype=torch.long)
                dummy_output = self.esmc(sequence_tokens=dummy_input)
                self.hidden_size = dummy_output.embeddings.shape[-1]
            except Exception:
                raise AttributeError("Não foi possível determinar hidden_size do modelo base ESMC.")

        self.attention_pool = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.esmc(sequence_tokens=input_ids)
        
        if hasattr(out, 'embeddings'):
            embeddings = out.embeddings
        elif hasattr(out, 'last_hidden_state'):
            embeddings = out.last_hidden_state
        else:
            raise AttributeError("Output do modelo base não contém embeddings")

        attn_logits = self.attention_pool(embeddings)
        
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1)
            # Use a smaller value like -1e4 instead of -1e9 for FP16 compatibility
            attn_logits = attn_logits.masked_fill(expanded_mask == 0, -1e4)

        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = (embeddings * attn_weights).sum(dim=1)
        
        x = self.dropout(pooled)
        logits = self.classifier(x)
        
        return SimpleNamespace(logits=logits)

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length: int = 1000):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long)
        )

def load_sequences(path, label):
    seqs, labs = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                seqs.append(s)
                labs.append(label)
    return seqs, labs

def train_epoch_amp(model, loader, optim, device, scheduler=None, gradient_accumulation_steps=4, scaler=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    
    optim.zero_grad()
    
    for i, (ids, mask, y) in enumerate(tqdm(loader, desc="Training")):
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        
        if scaler is not None:  # Precisão mista
            with autocast(device_type='cuda'):
                logits = model(ids, mask).logits
                loss = loss_fn(logits, y) / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:  # Precisão normal
            logits = model(ids, mask).logits
            loss = loss_fn(logits, y) / gradient_accumulation_steps
            loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0 or i == len(loader) - 1:
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
                
            # Atualiza o scheduler a cada batch efetivo (após gradient accumulation)
            if scheduler is not None:
                scheduler.step()
                
            optim.zero_grad()
        
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return correct/total, total_loss/len(loader)

def eval_model(model, loader, device, return_probs=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Evaluation"):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = model(ids, mask).logits
            loss = loss_fn(logits, y)
            
            # Calcula probabilidades
            probs = F.softmax(logits, dim=1)[:, 1]  # Classe positiva
            
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            if return_probs:
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
            
    metrics = {"accuracy": correct/total, "loss": total_loss/len(loader)}
    
    if return_probs:
        return metrics, np.array(all_probs), np.array(all_labels)
    return metrics

def find_best_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0, 1, 101):
        preds = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            
    return best_thr, best_f1

def calculate_metrics(y_true, y_prob, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.5,
    }

def _update_json(p: Path, entry: dict):
    data = {}
    if p.exists():
        data = json.loads(p.read_text())
    
    if "fold" in entry:
        data.setdefault("folds", []).append(entry)
    elif "epoch" in entry:
        data.setdefault("epochs", []).append(entry)
    elif "test" in entry:
        data["test"] = entry["test"] 
    elif "training_summary" in entry:
        data["training_summary"] = entry["training_summary"]
    
    p.write_text(json.dumps(data, indent=4))

class EarlyStopping:
    """Early stopping para interromper o treinamento quando a métrica monitorada não melhora."""
    def __init__(self, 
                 patience: int = 3, 
                 min_delta: float = 0.0, 
                 monitor: str = 'val_f1', 
                 mode: str = 'max',
                 verbose: bool = True):
        """
        Args:
            patience: Número de épocas sem melhoria para interromper
            min_delta: Mínima mudança que conta como melhoria
            monitor: Métrica para monitorar ('val_loss', 'val_f1', etc)
            mode: Direção da otimização ('min' para loss, 'max' para f1, accuracy, etc)
            verbose: Se True, imprime mensagens
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Inicializa valor comparativo baseado no modo
        self.best_score = float('-inf') if mode == 'max' else float('inf')
    
    def __call__(self, current: float, epoch: int, model: nn.Module, path: str) -> bool:
        """
        Decide se deve parar o treinamento e salva o melhor modelo.
        
        Args:
            current: Valor atual da métrica monitorada
            epoch: Época atual
            model: Modelo PyTorch atual
            path: Caminho para salvar o melhor modelo
            
        Returns:
            True se deve parar o treinamento, False caso contrário
        """
        score = current
        
        if self.mode == 'min':
            # Para métricas onde menor é melhor (ex: loss)
            improvement = self.best_score - score > self.min_delta
        else:
            # Para métricas onde maior é melhor (ex: accuracy, f1)
            improvement = score - self.best_score > self.min_delta
        
        if improvement:
            if self.verbose:
                print(f"Melhoria na métrica {self.monitor}: {self.best_score:.5f} -> {score:.5f}")
            self.best_score = score
            self.counter = 0
            
            # Salva o melhor modelo
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)
            
            if self.verbose:
                print(f"Salvando melhor modelo na época {epoch} em {path}")
                
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} épocas sem melhoria")
                
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"EarlyStopping ativado na época {epoch}")
                return True
                
            return False

def train_with_kfold(
    sequences: List[str], 
    labels: List[int], 
    n_folds: int = 5, 
    max_epochs: int = 15,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    patience: int = 3,
    use_amp: bool = True,
    scheduler_type: str = 'cosine',  # 'linear', 'cosine' ou 'one_cycle'
    warmup_steps_fraction: float = 0.1,
    device_str: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, float]]:

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    print(f"Dispositivo: {device}")
    print(f"Batch size efetivo: {batch_size * gradient_accumulation_steps}")
    
    # Diretório para o experimento atual
    run_dir = Path("model") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    model_dir = run_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "model_metrics.json"
    
    # Prepara dataset completo
    full_dataset = ProteinDataset(sequences, labels, tokenizer)
    
    # Validação cruzada estratificada
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Métricas para cada fold
    fold_metrics = []
    
    # Para média das métricas e escolha do melhor fold
    best_fold = -1
    best_val_f1 = -float('inf')
    
    # Loop para cada fold
    for fold, (train_idx, val_idx) in enumerate(kfold.split(sequences, labels)):
        print(f"\n{'='*20} Fold {fold+1}/{n_folds} {'='*20}")
        
        fold_start_time = time.time()
        
        # Inicializa um novo modelo para este fold
        model = ESMCForSequenceClassification().to(device)
        
        # Multi-GPU se disponível
        if torch.cuda.device_count() > 1:
            print(f"Usando {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # DataLoaders
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            full_dataset,
            batch_size=batch_size * 2,  # Pode usar batch maior na validação
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # Configuração do otimizador
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Configura total de steps para o scheduler
        num_training_steps = len(train_loader) * max_epochs // gradient_accumulation_steps
        warmup_steps = int(num_training_steps * warmup_steps_fraction)
        
        # Configuração do scheduler
        if scheduler_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'one_cycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=num_training_steps,
                pct_start=warmup_steps_fraction,
                anneal_strategy='cos'
            )
        else:
            scheduler = None
        
        # Configura scaler para mixed precision
        scaler = GradScaler() if use_amp else None
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            monitor='val_f1',
            mode='max',
            verbose=True
        )
        
        # Caminho para salvar o melhor modelo deste fold
        best_model_path = model_dir / f"best_model_fold_{fold+1}.pt"
        
        # Treinamento
        fold_history = []
        for epoch in range(1, max_epochs + 1):
            # Treino
            train_acc, train_loss = train_epoch_amp(
                model, train_loader, optimizer, device, 
                scheduler, gradient_accumulation_steps, scaler
            )
            
            # Validação e métricas detalhadas
            val_metrics, val_probs, val_true = eval_model(model, val_loader, device, return_probs=True)
            
            # Calcula threshold ideal para F1
            best_thr, best_f1 = find_best_threshold(val_true, val_probs)
            
            # Calcula métricas completas com o threshold ideal
            full_val_metrics = calculate_metrics(val_true, val_probs, threshold=best_thr)
            
            # Adiciona loss e accuracy das funções de treino/validação
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": full_val_metrics["f1"],
                "val_precision": full_val_metrics["precision"],
                "val_recall": full_val_metrics["recall"],
                "val_threshold": best_thr,
                "val_mcc": full_val_metrics["mcc"],
                "val_auc": full_val_metrics["auc"]
            }
            
            # Adiciona ao histórico e salva no arquivo
            fold_history.append(epoch_metrics)
            _update_json(metrics_file, {
                "fold": fold + 1,
                "epoch": epoch,
                **epoch_metrics
            })
            
            print(f"Fold {fold+1}, Epoch {epoch}/{max_epochs} — "
                  f"train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}, "
                  f"val_f1={full_val_metrics['f1']:.4f} (thr={best_thr:.2f})")
            
            # Early stopping baseado na métrica F1
            if early_stopping(full_val_metrics["f1"], epoch, model, best_model_path):
                print(f"Early stopping ativado na época {epoch}")
                break
        
        # Carrega o melhor modelo deste fold para avaliação final
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(best_model_path))
        else:
            model.load_state_dict(torch.load(best_model_path))
        
        # Avaliação final neste fold
        final_val_metrics, final_val_probs, final_val_true = eval_model(
            model, val_loader, device, return_probs=True
        )
        
        # Encontra o melhor threshold para F1 final
        final_best_thr, final_best_f1 = find_best_threshold(final_val_true, final_val_probs)
        
        # Calcula métricas completas com o threshold ideal
        fold_final_metrics = calculate_metrics(final_val_true, final_val_probs, threshold=final_best_thr)
        
        fold_time = time.time() - fold_start_time
        
        # Adiciona métricas finais do fold
        fold_summary = {
            "fold": fold + 1,
            "best_epoch": len(fold_history),
            "threshold": final_best_thr,
            "training_time": fold_time,
            "metrics": fold_final_metrics
        }
        
        fold_metrics.append(fold_final_metrics)
        _update_json(metrics_file, fold_summary)
        
        # Verifica se este é o melhor fold até agora
        if fold_final_metrics["f1"] > best_val_f1:
            best_val_f1 = fold_final_metrics["f1"]
            best_fold = fold
            
            # Copia o melhor modelo deste fold para o melhor modelo geral
            import shutil
            best_overall_model_path = model_dir / "best_model.pt"
            shutil.copy(best_model_path, best_overall_model_path)
    
    # Calcula médias das métricas de todos os folds
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        if metric != "threshold":  # Não faz sentido calcular média de threshold
            values = [fold[metric] for fold in fold_metrics]
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            avg_metrics[f"std_{metric}"] = np.std(values)
    
    # Salva resumo da validação cruzada
    cv_summary = {
        "training_summary": {
            "n_folds": n_folds,
            "best_fold": best_fold + 1,
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * gradient_accumulation_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_epochs": max_epochs,
                "early_stopping_patience": patience,
                "scheduler": scheduler_type,
                "warmup_fraction": warmup_steps_fraction
            }
        }
    }
    
    _update_json(metrics_file, cv_summary)
    
    # Carrega e retorna o melhor modelo
    best_model = ESMCForSequenceClassification().to(device)
    best_model.load_state_dict(torch.load(model_dir / "best_model.pt"))
    
    return best_model, avg_metrics, run_dir

def main():
    # Parâmetros
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hiperparâmetros otimizados
    batch_size = 4
    gradient_accumulation_steps = 8  # Aumentado para batch efetivo de 32
    learning_rate = 2e-5  # Taxa recomendada
    scheduler_type = 'cosine'  # Scheduler com warmup
    warmup_steps_fraction = 0.1
    n_folds = 5
    max_epochs = 5
    patience = 2
    use_amp = True  # Precisão mista
    
    # Carregamento de dados
    print("Carregando dados...")
    pos_seq, pos_lab = load_sequences("simalergenico.txt", 1)
    neg_seq, neg_lab = load_sequences("naoalergenico.txt", 0)
    sequences, labels = pos_seq + neg_seq, pos_lab + neg_lab
    
    # Dados de teste (mantidos completamente separados da validação cruzada)
    test_pos, test_pos_lab = load_sequences("test_sim.txt", 1)
    test_neg, test_neg_lab = load_sequences("test_nao.txt", 0)
    test_seqs, test_labs = test_pos + test_neg, test_pos_lab + test_neg_lab
    test_dataset = ProteinDataset(test_seqs, test_labs, tokenizer)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Total de sequências: {len(sequences)}")
    print(f"Positivas: {sum(labels)}, Negativas: {len(labels) - sum(labels)}")
    print(f"Conjunto de teste: {len(test_seqs)} sequências")
    
    # Treina com validação cruzada
    print("\n===== Iniciando treino com validação cruzada =====")
    best_model, avg_metrics, run_dir = train_with_kfold(
        sequences=sequences,
        labels=labels,
        n_folds=n_folds,
        max_epochs=max_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        patience=patience,
        use_amp=use_amp,
        scheduler_type=scheduler_type,
        warmup_steps_fraction=warmup_steps_fraction,
        device_str=None  # Auto-detecta
    )
    
    # Avaliação no conjunto de teste
    print("\n===== Avaliando no conjunto de teste =====")
    test_metrics, test_probs, test_true = eval_model(best_model, test_loader, device, return_probs=True)
    
    # Encontra threshold ideal no conjunto de validação
    best_thr = avg_metrics.get("avg_threshold", 0.5)
    
    # Calibra threshold também no teste para comparação
    test_best_thr, _ = find_best_threshold(test_true, test_probs)
    
    # Avalia usando ambos thresholds para comparação
    test_metrics_val_thr = calculate_metrics(test_true, test_probs, threshold=best_thr)
    test_metrics_test_thr = calculate_metrics(test_true, test_probs, threshold=test_best_thr)
    
    print(f"Métricas de teste (threshold da validação={best_thr:.3f}):")
    for k, v in test_metrics_val_thr.items():
        if k != "threshold":
            print(f"  {k}: {v:.4f}")
    
    print(f"\nMétricas de teste (threshold calibrado no teste={test_best_thr:.3f}):")
    for k, v in test_metrics_test_thr.items():
        if k != "threshold":
            print(f"  {k}: {v:.4f}")
    
    # Salva resultados do teste
    metrics_file = run_dir / "model_metrics.json"
    _update_json(metrics_file, {"test": {
        "with_validation_threshold": test_metrics_val_thr,
        "with_calibrated_threshold": test_metrics_test_thr
    }})
    
    # Salva tokenizer e run_info.json
    tokenizer.save_pretrained(run_dir / "tokenizer")
    run_info = {
        "seed": SEED,
        "best_validation_threshold": best_thr,
        "test_threshold": test_best_thr,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "scheduler": scheduler_type,
        "early_stopping_patience": patience,
        "test_results": test_metrics_test_thr,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=4))
    print(f"Todos os resultados salvos em {run_dir}")

if __name__ == "__main__":
    main()