# eata.py
# ------------------------------------------------------------------
# Implementação autocontida do Efficient Anti-forgetting
# Test-time Adaptation (EATA).
# ------------------------------------------------------------------
from __future__ import annotations
from copy import deepcopy
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


# -------------------- utilidades numéricas ------------------------ #
@torch.jit.script
def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H(X) = - Σ p log p  (por amostra)."""
    p = logits.softmax(dim=1)
    return -(p * p.log()).sum(dim=1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """similaridade cosseno fila-a-fila para tensores 2-D."""
    return F.cosine_similarity(a, b, dim=1, eps=1e-8)


# --------------------------- EATA --------------------------------- #
class EATA(nn.Module):
    """
    Implementa todo o algoritmo de adaptação em tempo de teste
    descrito no paper ICML-22.

    • Apenas parâmetros affine das BatchNorms são atualizados.
    • Entropia minimizada em amostras filtradas (Sent × Sdiv).
    • Regularização Fisher para evitar esquecimento.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2.5e-4,
        momentum: float = 0.9,
        steps: int = 1,
        episodic: bool = False,
        #
        e_margin: float = 0.40 * math.log(1000),  # E0
        d_margin: float = 0.05,  # ε
        #
        fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        fisher_alpha: float = 2_000.0,
        #
        ema_alpha: float = 0.9,  # α da média móvel
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.model: nn.Module = self._configure_model(model).to(device)
        self.params, _ = self._collect_adaptable_params(self.model)

        self.optimizer = torch.optim.SGD(self.params, lr, momentum=momentum)
        self.device = torch.device(device)

        # Hiper-parâmetros
        self.steps = max(1, steps)
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.ema_alpha = ema_alpha

        # Fisher
        self.fishers = fishers
        self.fisher_alpha = fisher_alpha

        # Estado online
        self._prob_avg: Optional[torch.Tensor] = None
        self.episodic = episodic
        # cria snapshot **apenas** se for preciso reiniciar a cada episódio
        self._model_state, self._opt_state = (None, None)
        if self.episodic:
            self._model_state, self._opt_state = self._snapshot_to_cpu()

    # ----------------------------------------------------------
    def _snapshot_to_cpu(self):
        # guarda pesos no CPU para não duplicar memória da GPU
        model_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        optim_cpu = deepcopy(self.optimizer.state_dict())
        for k, v in optim_cpu.items():
            if torch.is_tensor(v):
                optim_cpu[k] = v.detach().cpu()
        return model_cpu, optim_cpu

    # ------------------------------------------------------------------ #
    # ---------------  API pública chamada pelo Trainer  --------------- #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run_eata(self, loader):
        """
        Aplica EATA batch-a-batch no loader de teste.
        Apenas os parâmetros affine das camadas normais terão gradientes.
        """
        self.model.train()
        ys, preds, probs = [], [], []

        for ids, mask, y in tqdm(loader, desc="Test with EATA"):
            ids, mask, y = ids.to(self.device), mask.to(self.device), y.to(self.device)

            with torch.set_grad_enabled(True):
                out = self._forward_and_adapt(ids)

            with torch.no_grad():
                prob = F.softmax(out, 1)[:, 1]
                ys.extend(y.cpu().numpy())
                preds.extend(out.argmax(1).cpu().numpy())
                probs.extend(prob.cpu().numpy())

            del out, prob, ids, mask, y
            torch.cuda.empty_cache()

        return ys, preds, probs

    # ------------------------------------------------------------------ #
    # -------------------------  Internals  ---------------------------- #
    # ------------------------------------------------------------------ #
    def _forward_and_adapt(self, ids: torch.Tensor, mask=None) -> torch.Tensor:
        """
        ➊ Executa 'self.steps' updates de entropia no mesmo lote
        ➋ Depois faz um forward SEM gradiente para obter os logits finais
        (economiza memória porque o grafo de cada passo é liberado antes
        de criar o próximo).
        """
        for t in range(self.steps):
            # ---------- forward com grad ----------
            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(ids, mask)

            # ---------- filtros ----------
            entropy = softmax_entropy(logits)
            keep = entropy < self.e_margin
            if keep.sum() == 0:
                # Nada a adaptar → saia do loop
                continue

            logits_k = logits[keep]
            entropy_k = entropy[keep]

            if self._prob_avg is not None:
                cos_sim = cosine(
                    logits_k.softmax(1), self._prob_avg.unsqueeze(0).expand_as(logits_k)
                )
                keep_div = torch.abs(cos_sim) < self.d_margin

                if keep_div.sum() == 0:  # redundantes
                    continue
                logits_k, entropy_k = logits_k[keep_div], entropy_k[keep_div]

            # ---------- perda ----------
            coeff = torch.exp(-(entropy_k - self.e_margin))
            loss = (coeff * entropy_k).mean()
            if self.fishers:
                loss += self._ewc()

            # ---------- back-prop ----------
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # ---------- EMA + limpeza ----------
            self._update_prob_avg(logits_k.softmax(1).detach())
            # libera tudo que ainda referencia o grafo
            del logits, logits_k, entropy, entropy_k, loss
            torch.cuda.empty_cache()

        # ---------- forward final sem grad (leve em memória) ----------
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.device.type == "cuda"
        ):
            final_logits = self.model(ids, mask)

        return final_logits

    # -----------------  componentes auxiliares  ----------------------- #
    def _ewc(self) -> torch.Tensor:
        ewc_loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fishers:
                omega, init_val = self.fishers[n]
                ewc_loss += (omega * (p - init_val).pow(2)).sum()
        return self.fisher_alpha * ewc_loss

    def _update_prob_avg(self, probs: torch.Tensor):
        mean_probs = probs.mean(0)  # (C,)
        if self._prob_avg is None:
            self._prob_avg = mean_probs
        else:
            self._prob_avg = (
                self.ema_alpha * self._prob_avg + (1 - self.ema_alpha) * mean_probs
            )

    # ------------------  configuração de BN  -------------------------- #
    @staticmethod
    def _collect_adaptable_params(model: nn.Module):
        """
        Coleta parâmetros adaptáveis de camadas normalizadoras: BatchNorm* e LayerNorm.
        Retorna lista de parâmetros e seus nomes.
        """
        params, names = [], []
        norm_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
        for nm, m in model.named_modules():
            if isinstance(m, norm_classes):
                for pn, p in m.named_parameters():
                    if pn in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{pn}")
        return params, names

    @staticmethod
    def _configure_model(model: nn.Module):
        model.train()
        model.requires_grad_(False)
        norm_classes = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
        for m in model.modules():
            if isinstance(m, norm_classes):

                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model

    def reset(self):
        """Restaura pesos/opt se modo 'episodic'."""
        if self._model_state is None:
            return
        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._opt_state)
        self._prob_avg = None
