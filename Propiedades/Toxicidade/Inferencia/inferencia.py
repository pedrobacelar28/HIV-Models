# -*- coding: utf-8 -*-
"""
ESMC ‑ Binary classification — **inference‑only script**

1. Ajuste `RUN_DIR` abaixo para apontar para a pasta `run_YYYYmmdd_HHMMSS`
   gerada durante o treinamento (ela deve conter `model/pytorch_model.bin`
   e `tokenizer/`).
2. Ajuste, se necessário, o caminho dos arquivos‑teste:
      simsoltest.txt  (rótulo 1)
      naosoltest.txt  (rótulo 0)
3. Execute:  python infer_esmc.py
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
#  Imports
# --------------------------------------------------------------------------------------
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, amp
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

# --------------------------------------------------------------------------------------
#  Reproducibility (seed apenas para dataloader order, aqui irrelevante p/ pesos)
# --------------------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)

# --------------------------------------------------------------------------------------
#  ESM‑C: base + tokenizer
# --------------------------------------------------------------------------------------
from esm.pretrained import load_local_model
from esm.tokenization import get_esmc_model_tokenizers

# --------------------------------------------------------------------------------------
#  Modelo de classificação (mesmo do treinamento)
# --------------------------------------------------------------------------------------
class EsmClassificationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense   = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        self.out_proj= nn.Linear(cfg.hidden_size, cfg.num_labels)
    def forward(self, feats):
        x = feats[:, 0, :]          # <cls>
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)

class ESMCForSequenceClassification(nn.Module):
    def __init__(self, num_labels=2, base_model="esmc_300m", dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.esmc = load_local_model(base_model)
        cfg = SimpleNamespace(hidden_size=self.esmc.embed.embedding_dim,
                              hidden_dropout_prob=dropout,
                              num_labels=num_labels)
        self.classifier = EsmClassificationHead(cfg)
    def forward(self, input_ids, attention_mask=None, return_dict=True):
        outs   = self.esmc(sequence_tokens=input_ids)
        logits = self.classifier(outs.embeddings)
        return SimpleNamespace(logits=logits) if return_dict else logits

# --------------------------------------------------------------------------------------
#  Dataset
# --------------------------------------------------------------------------------------
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length: int = 1400):
        self.sequences  = sequences
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length
    def __len__(self):  return len(self.sequences)
    def __getitem__(self, idx):
        seq   = self.sequences[idx]
        label = self.labels[idx]
        enc   = self.tokenizer(
            seq, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        ids   = enc["input_ids"].squeeze(0)
        mask  = enc["attention_mask"].squeeze(0)
        return ids, mask, torch.tensor(label, dtype=torch.long)

# --------------------------------------------------------------------------------------
#  Utils
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

def metrics(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for ids, mask, y in tqdm(loader, desc="Inference"):
            ids, mask = ids.to(device), mask.to(device)
            with amp.autocast(device_type=device.type,
                              dtype=torch.float16,
                              enabled=(device.type == "cuda")):
                out = model(ids, mask).logits
            prob = F.softmax(out, 1)[:, 1]
            pred = out.argmax(1)
            ys.extend(y.numpy())
            preds.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())
    y_true, y_pred, y_prob = map(np.array, (ys, preds, probs))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":     accuracy_score(y_true, y_pred),
        "precision":    precision_score(y_true, y_pred, zero_division=0),
        "recall":       recall_score(y_true, y_pred, zero_division=0),
        "specificity":  tn / (tn + fp) if (tn + fp) else 0,
        "f1":           f1_score(y_true, y_pred, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "auc":          roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }

def load_saved(run_dir: Path, device):
    print(f"Loading model from {run_dir}")

    # 1)  Tokenizador – recriado diretamente
    tok = get_esmc_model_tokenizers()            # ← sem .from_pretrained()

    # 2)  Modelo + pesos
    mdl  = ESMCForSequenceClassification().to(device)
    state = torch.load(run_dir / "model" / "pytorch_model.bin",
                       map_location=device)
    mdl.load_state_dict(state)
    mdl.eval()
    return mdl, tok

# --------------------------------------------------------------------------------------
#  MAIN
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # -- 1. Caminhos ---------------------------------------------------
    RUN_DIR = Path("/scratch/pedro.bacelar/HIV-Models/Propiedades/Solubilidade/Modelos/model/run_20250421_233905")   # <–– editar aqui
    TEST_POS_TXT = "/scratch/pedro.bacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Testes/sim_eSol.txt"
    TEST_NEG_TXT = "/scratch/pedro.bacelar/HIV-Models/Propiedades/Solubilidade/Dataset/Testes/nao_eSol.txt"

    # -- 2. Dispositivo ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 3. Carrega modelo/ tokenizer ---------------------------------
    model, tokenizer = load_saved(RUN_DIR, device)

    # -- 4. Carrega conjunto de teste ---------------------------------
    t_pos, t_lab_pos = load_sequences(TEST_POS_TXT, 1)
    t_neg, t_lab_neg = load_sequences(TEST_NEG_TXT, 0)
    test_ds = ProteinDataset(t_pos + t_neg, t_lab_pos + t_lab_neg, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # -- 5. Inferência -------------------------------------------------
    test_metrics = metrics(model, test_loader, device)
    print("\n=== Test Metrics ===")
    for k, v in test_metrics.items():
        print(f"{k:12s}: {v:.4f}")

    # -- 6. Anexa no JSON da run --------------------------------------
    metrics_file = RUN_DIR / "model_metrics.json"
    data = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            data = json.load(f)
    data["test"] = test_metrics
    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nResultados gravados em {metrics_file}")
