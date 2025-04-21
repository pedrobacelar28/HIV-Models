#!/usr/bin/env python3
import os
import json
import random
import datetime
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

from esm.pretrained import load_local_model
from esm.tokenization import get_esmc_model_tokenizers

# ───────────────────────────── 1. Reproducibilidade ─────────────────────────────
SEED = 42
def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed()

# ───────────────────────────── 2. Dataset util ─────────────────────────────
class ProteinDataset(Dataset):
    def __init__(self, seqs, labels, tok, max_len=1780):
        self.seqs, self.labels, self.tok, self.max_len = seqs, labels, tok, max_len
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, i):
        enc = self.tok(self.seqs[i],
                       truncation=True,
                       padding='max_length',
                       max_length=self.max_len,
                       return_tensors='pt')
        return (
            enc['input_ids'].squeeze(0),
            enc['attention_mask'].squeeze(0),
            torch.tensor(self.labels[i], dtype=torch.long)
        )

def load_sequences(path, label):
    with open(path) as f:
        seqs = [l.strip() for l in f if l.strip()]
    return seqs, [label] * len(seqs)

# ───────────────────────────── 3. Carrega dados ─────────────────────────────
train_pos, y_train_pos = load_sequences('toxic.txt', 1)
train_neg, y_train_neg = load_sequences('non_toxic.txt', 0)
val_pos,   y_val_pos   = load_sequences('toxic_val.txt', 1)
val_neg,   y_val_neg   = load_sequences('non_toxic_val.txt', 0)
test_pos,  y_test_pos  = load_sequences('toxic_test.txt', 1)
test_neg,  y_test_neg  = load_sequences('non_toxic_test.txt', 0)

train_seqs, y_train = train_pos + train_neg, y_train_pos + y_train_neg
val_seqs,   y_val   = val_pos   + val_neg,   y_val_pos   + y_val_neg
test_seqs,  y_test  = test_pos  + test_neg,  y_test_pos  + y_test_neg

# ───────────────────────────── 4. Tokenizer & modelo ─────────────────────────────
BASE_MODEL = "esmc_300m"
tokenizer = get_esmc_model_tokenizers()

class ESMMeanClassifier(nn.Module):
    def __init__(self, base, num_labels=2, p=0.3):
        super().__init__()
        # usa o esmC carregado localmente
        self.esm = load_local_model(base)
        h = self.esm.embed.embedding_dim
        self.dropout = nn.Dropout(p)
        self.cls = nn.Linear(h, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.esm(sequence_tokens=input_ids)
        hs = out.embeddings  # [B, L, D] embeddings do esmC
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (hs * mask).sum(1) / (mask.sum(1).clamp_min(1e-9))
        else:
            pooled = hs.mean(1)
        logits = self.cls(self.dropout(pooled))
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)

model = ESMMeanClassifier(BASE_MODEL)

# ───────────────────────────── 5. DataLoaders ─────────────────────────────
BS = 2
train_loader = DataLoader(ProteinDataset(train_seqs, y_train, tokenizer),
                          batch_size=BS, shuffle=True)
val_loader   = DataLoader(ProteinDataset(val_seqs,   y_val,   tokenizer),
                          batch_size=BS)
test_loader  = DataLoader(ProteinDataset(test_seqs,  y_test,  tokenizer),
                          batch_size=BS)

# ───────────────────────────── 6. Otimizador/scheduler ─────────────────────────────
optimizer   = AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
EPOCHS      = 10
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(optimizer,
                                              int(0.002 * total_steps),
                                              total_steps)

# ───────────────────────────── 7. Funções de treino ─────────────────────────────
def train_epoch(model, dl, optim, sch, device):
    model.train()
    tl, corr, n = 0, 0, 0
    for ids, msk, lbl in tqdm(dl, desc='Train'):
        ids, msk, lbl = ids.to(device), msk.to(device), lbl.to(device)
        optim.zero_grad()
        out = model(ids, attention_mask=msk, labels=lbl)
        out.loss.backward()
        optim.step()
        sch.step()
        tl += out.loss.item()
        corr += (out.logits.argmax(1) == lbl).sum().item()
        n += lbl.size(0)
    return corr / n, tl / len(dl)

# ───────────────────────────── 8. Métricas util ─────────────────────────────
def score(y, p, probs):
    acc  = accuracy_score(y, p)
    prec = precision_score(y, p, zero_division=0)
    rec  = recall_score(y, p, zero_division=0)
    f1   = f1_score(y, p, zero_division=0)
    mcc  = matthews_corrcoef(y, p)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if tn + fp else 0
    auc  = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else .5
    return dict(accuracy=acc, precision=prec, recall=rec,
                specificity=spec, f1=f1, mcc=mcc, auc=auc)

# ───────────────────────────── 9. Avaliação simples ─────────────────────────────
@torch.no_grad()
def evaluate_plain(model, dl, device):
    model.eval()
    y, p, pr, total_loss = [], [], [], 0
    for ids, msk, lbl in dl:
        ids, msk, lbl = ids.to(device), msk.to(device), lbl.to(device)
        out = model(ids, attention_mask=msk, labels=lbl)
        logits = out.logits
        loss = out.loss

        probs = torch.softmax(logits, 1)[:, 1]
        y.extend(lbl.cpu().numpy())
        p.extend(logits.argmax(1).cpu().numpy())
        pr.extend(probs.cpu().numpy())
        total_loss += loss.item()

    metrics = score(np.array(y), np.array(p), np.array(pr))
    metrics["loss"] = total_loss / len(dl)
    return metrics




# ───────────────────────────── 10. Test‑Time Adaptation ─────────────────────────────


# ⚠️  SEM @torch.no_grad()  → precisa de gradientes
# ───────────────────────────── 10. Test‑Time Adaptation ─────────────────────────────
def enable_tta_params(model):
    for p in model.parameters(): p.requires_grad = False
    for mod in model.modules():
        if isinstance(mod, nn.LayerNorm):
            mod.train()
            for p in mod.parameters(): p.requires_grad = True
    linear_candidates = [(n,m) for n,m in model.named_modules()
                         if isinstance(m, nn.Linear) and not n.startswith('esm.')]
    if not linear_candidates:
        raise RuntimeError("Nenhuma Linear de cabeça encontrada!")
    for p in sorted(linear_candidates, key=lambda x: len(x[0]))[-1][1].parameters():
        p.requires_grad = True

# ⚠️  SEM @torch.no_grad()  → precisa de gradientes
def evaluate_tta(model, dl, device, lr=1e-4):
    model_tta = copy.deepcopy(model).to(device)
    enable_tta_params(model_tta)
    opt = SGD(filter(lambda p: p.requires_grad, model_tta.parameters()), lr=lr, momentum=0.9)

    y, p, pr = [], [], []
    model_tta.train()
    for ids, msk, lbl in dl:
        ids, msk = ids.to(device), msk.to(device)

        logits = model_tta(ids, attention_mask=msk).logits
        loss   = -torch.mean(torch.sum(torch.softmax(logits,1) *
                                       torch.log_softmax(logits,1), 1))
        opt.zero_grad(); loss.backward(); opt.step()

        probs = torch.softmax(logits,1)[:,1].detach()
        y.extend(lbl.numpy())
        p.extend(logits.argmax(1).cpu().numpy())
        pr.extend(probs.cpu().numpy())
    return score(np.array(y), np.array(p), np.array(pr))

# ───────────────────────────── 11. Loop principal ─────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

run_dir = os.path.join('model', f'run_{datetime.datetime.now():%Y%m%d_%H%M%S}')
os.makedirs(run_dir, exist_ok=True)
metrics_path = os.path.join(run_dir,'metrics.json')

def save_metrics(tag, d):
    data = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    data[tag] = d
    json.dump(data, open(metrics_path,'w'), indent=4)

for ep in range(1, EPOCHS+1):
    tr_acc, tr_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_plain = evaluate_plain(model, val_loader, device)
    val_tta   = evaluate_tta(model, val_loader, device)
    save_metrics(f'epoch_{ep}', {"train":{"loss":tr_loss,"accuracy":tr_acc},
                                 "plain_val":val_plain, "tta_val":val_tta})
    print(f"Ep {ep} | Train loss: {tr_loss:.4f} | Val loss: {val_plain['loss']:.4f} "
      f"| F1 (plain): {val_plain['f1']:.3f} | F1 (tta): {val_tta['f1']:.3f}")


# ───────────────────────────── 12. Teste final ─────────────────────────────
test_plain = evaluate_plain(model, test_loader, device)
test_tta   = evaluate_tta(model, test_loader, device)
save_metrics('test', {"plain_test":test_plain, "tta_test":test_tta})
print("Test (plain):", test_plain)
print("Test (tta)  :", test_tta)

# ───────────────────────────── 13. Salvar artefatos ─────────────────────────────
# ────────────────────────────────────────────────────────────────
# 14. Salvar artefatos
# ────────────────────────────────────────────────────────────────
# cria a pasta (caso não exista)
model_dir = os.path.join(run_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# 1) pesos do modelo (inclui backbone + classifier)
torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

# 2) config do backbone — opcional, mas útil para recriar o modelo
model.esm.config.to_json_file(os.path.join(model_dir, "config.json"))

# 3) tokenizer (continua igual)
tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))