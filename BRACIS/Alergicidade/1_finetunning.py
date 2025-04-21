# -*- coding: utf-8 -*-
"""
Finetuning ESMC para classificação binária de sequências,
com attention pooling e threshold calibrado
"""

from __future__ import annotations

import os
import json
import random
import datetime
from pathlib import Path
from types import SimpleNamespace
from torch.amp import autocast, GradScaler

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
)

# ESM‑C imports
from esm.pretrained import load_local_model
from esm.tokenization import get_esmc_model_tokenizers

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

def train_epoch_amp(model, loader, optim, device, gradient_accumulation_steps=4, scaler=None):
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
            optim.zero_grad()
        
        batch_loss = loss.item() * gradient_accumulation_steps
        total_loss += batch_loss
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    return correct/total, total_loss/len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Validation"):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = model(ids, mask).logits
            loss = loss_fn(logits, y)
            
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    return correct/total, total_loss/len(loader)

def find_best_threshold(model, loader, device):
    model.eval()
    ys, probs = [], []
    
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask = ids.to(device), mask.to(device)
            out = model(ids, mask).logits
            p1 = F.softmax(out, dim=1)[:, 1]
            ys.extend(y.numpy())
            probs.extend(p1.cpu().numpy())
            
    y_true = np.array(ys)
    probs = np.array(probs)

    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0, 1, 101):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            
    return best_thr

def metrics(model, loader, device, threshold: float = 0.5):
    model.eval()
    ys, probs = [], []
    
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Testing"):
            ids, mask = ids.to(device), mask.to(device)
            out = model(ids, mask).logits
            p1 = F.softmax(out, dim=1)[:, 1]
            ys.extend(y.numpy())
            probs.extend(p1.cpu().numpy())

    y_true = np.array(ys)
    y_prob = np.array(probs)
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
    if "epoch" in entry:
        data.setdefault("epochs", []).append(entry)
    else:
        data["test"] = entry["test"]
    p.write_text(json.dumps(data, indent=4))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradient_accumulation_steps = 4
    use_amp = True  # Precisão mista

    # Carregamento de dados com prefetching
    pos_seq, pos_lab = load_sequences("simalergenico.txt", 1)
    neg_seq, neg_lab = load_sequences("naoalergenico.txt", 0)
    sequences, labels = pos_seq + neg_seq, pos_lab + neg_lab
    dataset = ProteinDataset(sequences, labels, tokenizer)
    
    idx = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idx, test_size=0.10, stratify=labels, random_state=SEED)
    
    # DataLoaders otimizados
    train_loader = DataLoader(
        dataset, 
        batch_size=4, 
        sampler=SubsetRandomSampler(train_idx),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset, 
        batch_size=4, 
        sampler=SubsetRandomSampler(val_idx),
        num_workers=2,
        pin_memory=True
    )

    # Teste
    test_pos, test_pos_lab = load_sequences("test_sim.txt", 1)
    test_neg, test_neg_lab = load_sequences("test_nao.txt", 0)
    test_seqs = test_pos + test_neg
    test_labs = test_pos_lab + test_neg_lab
    test_dataset = ProteinDataset(test_seqs, test_labs, tokenizer)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Setup modelo
    model = ESMCForSequenceClassification().to(device)
    
    # Multi-GPU se disponível
    if torch.cuda.device_count() > 1:
        print(f"Usando {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # Configura scaler para mixed precision
    scaler = GradScaler() if use_amp else None
    
    # Diretório de logs
    run_dir = Path("model") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "model_metrics.json"

    # Treino
    epochs = 8
    for ep in range(1, epochs+1):
        tr_acc, tr_loss = train_epoch_amp(model, train_loader, optim, device, 
                                        gradient_accumulation_steps, scaler)
        v_acc, v_loss = eval_epoch(model, val_loader, device)
        print(f"Epoch {ep}/{epochs} — loss={tr_loss:.4f} val_acc={v_acc:.4f}")
        _update_json(metrics_file, {
            "epoch": ep,
            "training": {"loss": tr_loss, "accuracy": tr_acc},
            "validation": {"loss": v_loss, "accuracy": v_acc},
        })

    # Calibra threshold na validação
    best_thr = find_best_threshold(model, val_loader, device)
    print(f"Best validation threshold for F1: {best_thr:.2f}")

    # Avalia no teste usando threshold calibrado
    test_metrics = metrics(model, test_loader, device, threshold=best_thr)
    test_metrics["threshold"] = best_thr
    print("Test metrics:", test_metrics)
    _update_json(metrics_file, {"test": test_metrics})

    # Salva modelo, tokenizer e run_info.json
    torch.save(model.state_dict(), run_dir / "model" / "pytorch_model.bin")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    run_info = {
        "seed": SEED,
        "best_threshold": best_thr,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": 4 * gradient_accumulation_steps,
        "test_results": test_metrics,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=4))
    print(f"Saved to {run_dir}")

if __name__ == "__main__":
    main()