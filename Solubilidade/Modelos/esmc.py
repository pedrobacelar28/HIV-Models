# -*- coding: utf-8 -*-
"""
Finetuning ESMC for binary protein‑sequence classification
---------------------------------------------------------
*Adicionada scheduler `get_linear_schedule_with_warmup`* para manter o mesmo
fluxo de aprendizagem do seu script ESM‑2.
A forma de salvar **modelo**, **tokenizer** e **métricas** continua análoga:
- métricas → `model_metrics.json` (épocas + bloco test)
- artefatos → `run_YYYYmmdd_HHMMSS/` com `model/pytorch_model.bin` + tokenizer + `run_info.json`.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
#  Imports & seed
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
from transformers import get_linear_schedule_with_warmup  # << NEW

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------------------------------
#  ESMC base + tokenizer
# --------------------------------------------------------------------------------------
from esm.pretrained import load_local_model
from esm.tokenization import get_esmc_model_tokenizers

tokenizer = get_esmc_model_tokenizers()

class ESMCForSequenceClassification(nn.Module):
    """Pooling + classifier em cima do ESMC."""
    def __init__(self, num_labels: int = 2, base_model_name: str = "esmc_300m"):
        super().__init__()
        self.esmc = load_local_model(base_model_name)
        self.hidden_size = self.esmc.embed.embedding_dim
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.esmc(sequence_tokens=input_ids)
        emb = out.embeddings  # [B,L,D]
        # ----- dentro de forward() -----
        emb = out.embeddings                         # [B, L, D]
        cls_vec = emb[:, 0, :]                       # token <cls> está em idx 0
        logits = self.classifier(self.dropout(cls_vec))

        '''pooled = (
            (emb * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(1, keepdim=True)+1e-8)
            if attention_mask is not None else emb.mean(1)
        )
        logits = self.classifier(self.dropout(pooled))'''
        return SimpleNamespace(logits=logits)

# --------------------------------------------------------------------------------------
#  Dataset
# --------------------------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(self, seqs, labs, tok, max_len: int = 850):
        self.seqs, self.labs, self.tok, self.max_len = seqs, labs, tok, max_len
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        enc = self.tok(self.seqs[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labs[idx], dtype=torch.long),
        )

# --------------------------------------------------------------------------------------
#  Data
# --------------------------------------------------------------------------------------

def load_sequences(path, label):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines, [label]*len(lines)

pos_seq, pos_lab = load_sequences("simB.txt", 1)
neg_seq, neg_lab = load_sequences("naoB.txt", 0)
seqs, labs = pos_seq+neg_seq, pos_lab+neg_lab

dataset = ProteinDataset(seqs, labs, tokenizer)
idx = list(range(len(dataset)))
train_idx, tmp_idx = train_test_split(idx, test_size=0.10, stratify=labs, random_state=SEED)
val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, stratify=[labs[i] for i in tmp_idx], random_state=SEED)

train_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(train_idx))
val_loader   = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(val_idx))
test_loader  = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(test_idx))

# --------------------------------------------------------------------------------------
#  Train / eval helpers (scheduler support)
# --------------------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    tot_loss = corr = tot = 0
    loss_fn = nn.CrossEntropyLoss()
    for ids, mask, y in tqdm(loader, desc="Training"):
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(ids, mask).logits
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()  # << NEW
        tot_loss += loss.item()
        corr += (logits.argmax(1) == y).sum().item()
        tot += y.size(0)
    return corr/tot, tot_loss/len(loader)

def eval_epoch(model, loader, device):
    model.eval(); loss_fn = nn.CrossEntropyLoss(); tot_loss=corr=tot=0
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Validation"):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = model(ids, mask).logits
            loss = loss_fn(logits, y)
            tot_loss += loss.item()
            corr += (logits.argmax(1)==y).sum().item(); tot += y.size(0)
    return corr/tot, tot_loss/len(loader)

def compute_metrics(model, loader, device):
    model.eval(); y_true=y_pred=y_prob=[]; y_true=[];y_pred=[];y_prob=[]
    with torch.no_grad():
        for ids,mask,y in tqdm(loader,desc="Testing"):
            ids,mask = ids.to(device),mask.to(device)
            out = model(ids,mask).logits
            y_true.extend(y.numpy()); y_pred.extend(out.argmax(1).cpu().numpy()); y_prob.extend(F.softmax(out,1)[:,1].cpu().numpy())
    y_true, y_pred, y_prob = map(np.array, (y_true,y_pred,y_prob))
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return dict(
        accuracy=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred,zero_division=0),
        recall=recall_score(y_true,y_pred,zero_division=0),
        specificity=tn/(tn+fp) if (tn+fp) else 0,
        f1=f1_score(y_true,y_pred,zero_division=0),
        mcc=matthews_corrcoef(y_true,y_pred),
        auc=roc_auc_score(y_true,y_prob) if len(np.unique(y_true))>1 else 0.5,
    )

# --------------------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMCForSequenceClassification().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 5  # epochs later
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.02*total_steps),
        num_training_steps=total_steps,
    )
    epochs = 5

    run_dir = Path("model") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "model_metrics.json"

    for ep in range(1, epochs+1):
        tr_acc, tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        v_acc, v_loss   = eval_epoch(model, val_loader, device)
        val_metrics = compute_metrics(model, val_loader, device)
        print(f"Epoch {ep}/{epochs}  loss={tr_loss:.4f}  val_acc={v_acc:.4f}")
        _update_json(metrics_path, {"epoch": ep, "training": {"loss": tr_loss, "accuracy": tr_acc}, "validation": {"loss": v_loss, **val_metrics}})

    test_metrics = compute_metrics(model, test_loader, device)
    print("Test metrics:", test_metrics)
    _update_json(metrics_path, {"test": test_metrics})

    # --- salvar artefatos -----------------------------------------------------------
    torch.save(model.state_dict(), run_dir / "model" / "pytorch_model.bin")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    with open(run_dir / "run_info.json", "w") as fh:
        json.dump({"seed": SEED, "test_results": test_metrics}, fh, indent=4)
    print(f"Artefatos salvos em {run_dir}")

# helper JSON

def _update_json(path: Path, entry: dict):
    data={}
    if path.exists():
        with open(path) as f: data=json.load(f)
    if "epoch" in entry:
        data.setdefault("epochs", []).append(entry)
    else:
        data["test"] = entry["test"]
    with open(path,"w") as f:
        json.dump(data,f,indent=4)

if __name__ == "__main__":
    main()
