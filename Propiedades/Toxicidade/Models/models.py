from esm.pretrained import load_local_model  # local weights registry
import torch
import torch.nn as nn


class ESMCForSequenceClassification(nn.Module):
    """
    Two-layer MLP head on top of a frozen ESM-C encoder.
    Weight initialisation:
        • Linear layers  : Kaiming-normal (fan_out, non-linearity='relu')
        • Biases         : 0
    """

    def __init__(
        self,
        num_labels: int = 2,
        base_model: str = "esmc_300m",
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # 1) backbone -----------------------------------------------------
        self.esmc = load_local_model(base_model)
        if freeze_backbone:
            for p in self.esmc.parameters():
                p.requires_grad_(False)

        d_model = self.esmc.embed.embedding_dim  # hidden size

        # 2) classification head -----------------------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )
        self._init_weights()  # <- custom initialisation

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)  # match backbone
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, mask=None):
        """
        x : LongTensor [B, L]  – token IDs already padded/trimmed
        mask is ignored (kept for API compatibility).
        """
        outs = self.esmc(sequence_tokens=x)
        cls = outs.embeddings[:, 0, :]  # <bos> representation
        logits = self.head(cls)  # [B, num_labels]
        return logits
