# -*- coding: utf-8 -*-
"""
Finetuning ESMC for binary protein‑sequence classification
---------------------------------------------------------
Fixed: wrapper now reads hidden dimension from ``esmc.embed.embedding_dim``
(instead of nonexistent ``d_model`` attribute).
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
#  Boiler‑plate & utils (identical to previous version except seed const)
# --------------------------------------------------------------------------------------
import os
import json
import random
import datetime
from pathlib import Path
from types import SimpleNamespace

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------------------------------
#  ESMC‑specific imports & lightweight wrapper
# --------------------------------------------------------------------------------------
from esm.pretrained import load_local_model  # local weights registry
from esm.tokenization import get_esmc_model_tokenizers

# ---- 1. Tokenizer --------------------------------------------------------------------

tokenizer = get_esmc_model_tokenizers()

# ---- 2. Base model & classification head --------------------------------------------

class ESMCForSequenceClassification(nn.Module):
    """Tiny wrapper that adds a pooling layer + classifier on top of ESMC."""

    def __init__(self, num_labels: int = 2, base_model_name: str = "esmc_300m"):
        super().__init__()
        # Load base model (already moved to correct device later)
        self.esmc = load_local_model(base_model_name)
        # Hidden size is the embedding dimension of the token embedding layer
        self.hidden_size = self.esmc.embed.embedding_dim
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.esmc(sequence_tokens=input_ids)
        embeddings = out.embeddings  # [B, L, D]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (embeddings * mask).sum(1) / (mask.sum(1) + 1e-8)
        else:
            pooled = embeddings.mean(1)
        logits = self.classifier(self.dropout(pooled))
        return SimpleNamespace(logits=logits)

# --------------------------------------------------------------------------------------
#  Dataset class (unchanged)
# --------------------------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length: int = 850):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        enc = self.tokenizer(seq, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# --------------------------------------------------------------------------------------
#  Helper to load sequences
# --------------------------------------------------------------------------------------

def load_sequences(path, label):
    seqs, labs = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                seqs.append(s)
                labs.append(label)
    return seqs, labs

# --------------------------------------------------------------------------------------
#  Data loading & splitting
# --------------------------------------------------------------------------------------
POS_TXT = "simB.txt"
NEG_TXT = "naoB.txt"

pos_seq, pos_lab = load_sequences(POS_TXT, 1)
neg_seq, neg_lab = load_sequences(NEG_TXT, 0)

sequences = pos_seq + neg_seq
labels = pos_lab + neg_lab

dataset = ProteinDataset(sequences, labels, tokenizer)

idx = list(range(len(dataset)))
train_idx, temp_idx = train_test_split(idx, test_size=0.10, stratify=labels, random_state=SEED)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=[labels[i] for i in temp_idx], random_state=SEED)

train_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(train_idx))
val_loader   = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(val_idx))
test_loader  = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(test_idx))

# --------------------------------------------------------------------------------------
#  Training / evaluation utilities (same logic)
# --------------------------------------------------------------------------------------

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss = correct = total = 0
    loss_fn = nn.CrossEntropyLoss()
    for ids, mask, y in tqdm(loader, desc="Training"):
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        optim.zero_grad()
        logits = model(ids, mask).logits
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct/total, total_loss/len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = correct = total = 0
    loss_fn = nn.CrossEntropyLoss()
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

def metrics(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Testing"):
            ids, mask = ids.to(device), mask.to(device)
            out = model(ids, mask).logits
            prob = F.softmax(out, 1)[:,1]
            pred = out.argmax(1)
            ys.extend(y.numpy())
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    y_true, y_pred, y_prob = np.array(ys), np.array(preds), np.array(probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn+fp) if (tn+fp) else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.5,
    }

# --------------------------------------------------------------------------------------
#  Train‑test driver
# --------------------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMCForSequenceClassification().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    epochs = 5

    run_dir = Path("model") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "model_metrics.json"

    for ep in range(1, epochs+1):
        tr_acc, tr_loss = train_epoch(model, train_loader, optim, device)
        v_acc, v_loss = eval_epoch(model, val_loader, device)
        val_metrics = metrics(model, val_loader, device)
        print(f"Epoch {ep}/{epochs} — loss={tr_loss:.4f} val_acc={v_acc:.4f}")
        _update_json(metrics_file, {"epoch": ep, "training": {"loss": tr_loss, "accuracy": tr_acc}, "validation": {"loss": v_loss, **val_metrics}})

    test_metrics = metrics(model, test_loader, device)
    print("Test:", test_metrics)
    _update_json(metrics_file, {"test": test_metrics})

    # Save
    torch.save(model.state_dict(), run_dir / "model" / "pytorch_model.bin")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    with open(run_dir / "run_info.json", "w") as f:
        json.dump({"seed": SEED, "test_results": test_metrics}, f, indent=4)
    print(f"Saved to {run_dir}")

# --- helper to update JSON -------------------------------------------------------------

def _update_json(p: Path, entry: dict):
    data = {}
    if p.exists():
        with open(p) as f:
            data = json.load(f)
    if "epoch" in entry:
        data.setdefault("epochs", []).append(entry)
    else:
        data["test"] = entry["test"]
    with open(p, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
