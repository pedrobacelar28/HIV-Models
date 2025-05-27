from esm.pretrained import load_local_model  # local weights registry
import torch
import torch.nn as nn
from transformers import EsmForSequenceClassification, EsmTokenizer


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
        pretrained_backbone_path: str = None,  # NOVO: caminho para pesos pré-treinados
    ):
        super().__init__()

        # 1) backbone -----------------------------------------------------
        self.esmc = load_local_model(base_model)
        
        # NOVO: Carregar pesos pré-treinados se fornecido
        if pretrained_backbone_path is not None:
            self._load_pretrained_backbone(pretrained_backbone_path)
        
        if freeze_backbone:
            for p in self.esmc.parameters():
                p.requires_grad_(False)

        d_model = self.esmc.embed.embedding_dim  # hidden size

        # 2) classification head -----------------------------------------
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels),
        )
        self._init_weights()  # <- custom initialisation

    # ------------------------------------------------------------------ #
    def _load_pretrained_backbone(self, pretrained_path: str):
        """
        Carrega pesos pré-treinados no backbone ESM-C
        """
        print(f"🔄 Carregando pesos pré-treinados: {pretrained_path}")
        
        try:
            pretrained_state = torch.load(pretrained_path, map_location='cpu')
            
            # Carregar apenas os pesos compatíveis
            model_state = self.esmc.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_state}
            
            missing_keys = set(model_state.keys()) - set(pretrained_state.keys())
            if missing_keys:
                print(f"⚠️  Chaves não encontradas no modelo pré-treinado: {missing_keys}")
            
            # Aplicar pesos
            model_state.update(pretrained_state)
            self.esmc.load_state_dict(model_state)
            
            print(f"✅ Pesos pré-treinados carregados com sucesso!")
            print(f"   📊 Carregadas {len(pretrained_state)} camadas")
            
        except Exception as e:
            print(f"❌ Erro ao carregar pesos pré-treinados: {e}")
            print(f"   🔄 Continuando com pesos originais do {self.esmc.__class__.__name__}")

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


class ESM2ForSequenceClassification(nn.Module):
    """
    Wrapper para modelos ESM2 usando transformers library.
    Com inicialização de pesos personalizada similar ao ESMC.
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        base_model: str = "facebook/esm2_t33_650M_UR50D",
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        # Carregar apenas o modelo base ESM2 (sem cabeça de classificação)
        from transformers import EsmModel
        self.esm = EsmModel.from_pretrained(base_model)
        
        # Obter dimensão do modelo
        if hasattr(self.esm.config, 'hidden_size'):
            d_model = self.esm.config.hidden_size
        else:
            d_model = 1280  # Default para ESM2 modelos padrão
        
        # Criar nossa própria cabeça de classificação (similar ao ESMC)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels),
        )
        
        # Aplicar inicialização personalizada na cabeça
        self._init_classification_head()
        
        if freeze_backbone:
            # Congelar todas as camadas do backbone
            for param in self.esm.parameters():
                param.requires_grad = False
    
    def _init_classification_head(self):
        """
        Inicialização personalizada da cabeça de classificação,
        similar ao que é feito no ESMC.
        """
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                # Usar inicialização truncated normal como no ESMC
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, mask=None):
        """
        x : LongTensor [B, L]  – token IDs already padded/trimmed
        mask : attention mask for padding
        """
        # Forward no modelo ESM2 base
        outputs = self.esm(input_ids=x, attention_mask=mask)
        
        # Extrair representação do token CLS (primeira posição)
        # outputs.last_hidden_state shape: [batch_size, seq_len, hidden_size]
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Aplicar cabeça de classificação
        logits = self.classifier(cls_representation)  # [batch_size, num_labels]
        
        return logits


def create_model(base_model: str, num_labels: int = 2, **kwargs):
    """
    Factory function para criar o modelo apropriado baseado no nome.
    
    Args:
        base_model: Nome do modelo base
        num_labels: Número de classes para classificação
        **kwargs: Argumentos adicionais para o modelo (incluindo pretrained_backbone_path)
    
    Returns:
        Modelo inicializado
    """
    if base_model.startswith("esmc"):
        return ESMCForSequenceClassification(
            num_labels=num_labels,
            base_model=base_model,
            **kwargs
        )
    elif base_model.startswith("facebook/esm2") or base_model.startswith("esm2"):
        # Garantir que temos o nome completo do modelo HuggingFace
        if not base_model.startswith("facebook/"):
            base_model = f"facebook/{base_model}"
        
        # Para ESM2, remover pretrained_backbone_path se presente (não suportado ainda)
        esm2_kwargs = {k: v for k, v in kwargs.items() if k != 'pretrained_backbone_path'}
        if 'pretrained_backbone_path' in kwargs:
            print("⚠️  pretrained_backbone_path não suportado para ESM2, ignorando...")
        
        return ESM2ForSequenceClassification(
            num_labels=num_labels,
            base_model=base_model,
            **esm2_kwargs
        )
    else:
        raise ValueError(f"Modelo não suportado: {base_model}")


def get_tokenizer(base_model: str):
    """
    Retorna o tokenizer apropriado para o modelo.
    
    Args:
        base_model: Nome do modelo base
    
    Returns:
        Tokenizer inicializado
    """
    if base_model.startswith("esmc"):
        from esm.tokenization import get_esmc_model_tokenizers
        return get_esmc_model_tokenizers()
    elif base_model.startswith("facebook/esm2") or base_model.startswith("esm2"):
        if not base_model.startswith("facebook/"):
            base_model = f"facebook/{base_model}"
        return EsmTokenizer.from_pretrained(base_model)
    else:
        raise ValueError(f"Tokenizer não suportado para modelo: {base_model}")
