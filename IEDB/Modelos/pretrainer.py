"""
Pré-treinador MLM (Masked Language Modeling) para ESM-C em epitopos
Estratégia: Mascarar aminoácidos aleatórios e treinar o modelo para prever os tokens originais
"""

from __future__ import annotations
import json, random, datetime, os
from pathlib import Path
from typing import Dict, Tuple, Any, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from models import create_model, get_tokenizer
from copy import deepcopy


class MLMDataset(Dataset):
    """Dataset para Masked Language Modeling"""
    
    def __init__(self, sequences: List[str], tokenizer, max_length: int = 1200, 
                 mlm_probability: float = 0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Obter tokens especiais
        if hasattr(tokenizer, 'mask_token_id'):
            self.mask_token_id = tokenizer.mask_token_id
        elif hasattr(tokenizer, '_token_to_id') and '<mask>' in tokenizer._token_to_id:
            self.mask_token_id = tokenizer._token_to_id['<mask>']
        else:
            # Para ESM-C, usar um valor padrão ou descobrir dinamicamente
            self.mask_token_id = 32  # Valor típico para <mask> em ESM-C
        
        # Descobrir range de tokens válidos para aminoácidos
        self._discover_valid_tokens()
    
    def _discover_valid_tokens(self):
        """Descobre quais tokens são válidos para mascaramento (aminoácidos)"""
        # Aminoácidos padrão
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                      'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        self.valid_tokens = set()
        
        # Para ESM-C
        if hasattr(self.tokenizer, '_token_to_id'):
            for aa in amino_acids:
                if aa in self.tokenizer._token_to_id:
                    self.valid_tokens.add(self.tokenizer._token_to_id[aa])
        # Para ESM2 (transformers)
        elif hasattr(self.tokenizer, 'vocab'):
            for aa in amino_acids:
                if aa in self.tokenizer.vocab:
                    self.valid_tokens.add(self.tokenizer.vocab[aa])
        else:
            # Fallback: assumir range típico de tokens de aminoácidos
            self.valid_tokens = set(range(4, 24))  # Range típico para ESM
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenizar sequência
        if hasattr(self.tokenizer, '__call__'):
            # ESM2 (transformers)
            encoded = self.tokenizer(
                sequence,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
        else:
            # ESM-C
            tokens = self.tokenizer.encode(sequence)
            # Pad ou truncar
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [0] * (self.max_length - len(tokens))  # 0 = pad
            
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == 0] = 0  # Mask padding tokens
        
        # Criar labels para MLM (inicialmente cópia de input_ids)
        labels = input_ids.clone()
        
        # Aplicar mascaramento
        masked_input_ids, labels = self._apply_masking(input_ids, labels, attention_mask)
        
        return masked_input_ids, attention_mask, labels
    
    def _apply_masking(self, input_ids, labels, attention_mask):
        """Aplica mascaramento MLM aos tokens"""
        masked_input_ids = input_ids.clone()
        
        # Só mascarar tokens válidos (aminoácidos) e que não sejam padding
        valid_positions = []
        for i, token_id in enumerate(input_ids):
            if (attention_mask[i] == 1 and  # Não é padding
                token_id.item() in self.valid_tokens):  # É aminoácido
                valid_positions.append(i)
        
        if len(valid_positions) == 0:
            # Se não há posições válidas, retornar sem mascaramento
            labels.fill_(-100)  # -100 é ignorado na loss
            return masked_input_ids, labels
        
        # Calcular quantos tokens mascarar
        num_to_mask = max(1, int(len(valid_positions) * self.mlm_probability))
        
        # Selecionar posições aleatórias para mascarar
        positions_to_mask = random.sample(valid_positions, num_to_mask)
        
        # Aplicar estratégia de mascaramento (80% <mask>, 10% random, 10% unchanged)
        for pos in positions_to_mask:
            rand = random.random()
            if rand < 0.8:
                # 80%: substituir por <mask>
                masked_input_ids[pos] = self.mask_token_id
            elif rand < 0.9:
                # 10%: substituir por token aleatório de aminoácido
                random_aa_token = random.choice(list(self.valid_tokens))
                masked_input_ids[pos] = random_aa_token
            # 10%: manter original (já está correto)
        
        # Configurar labels: -100 para posições não mascaradas
        labels[~torch.isin(torch.arange(len(labels)), torch.tensor(positions_to_mask))] = -100
        
        return masked_input_ids, labels


class ESMCForMLM(nn.Module):
    """Wrapper do ESM-C para Masked Language Modeling"""
    
    def __init__(self, base_model: str = "esmc_300m"):
        super().__init__()
        
        from esm.pretrained import load_local_model
        self.esmc = load_local_model(base_model)
        
        # Cabeça MLM usando as mesmas dimensões do modelo base
        d_model = self.esmc.embed.embedding_dim
        vocab_size = self.esmc.embed.num_embeddings
        
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # Inicializar pesos da cabeça MLM
        nn.init.normal_(self.mlm_head.weight, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)
    
    def forward(self, input_ids, labels=None):
        # Forward pass pelo modelo base
        outputs = self.esmc(sequence_tokens=input_ids)
        
        # Obter representações de todos os tokens
        hidden_states = outputs.embeddings  # [batch_size, seq_len, hidden_size]
        
        # Aplicar cabeça MLM
        prediction_scores = self.mlm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # ignora -100 automaticamente
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), 
                           labels.view(-1))
        
        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': hidden_states
        }


class PretrainerESMC:
    """Pré-treinador MLM para ESM-C"""
    
    def __init__(
        self,
        sequences: List[str],
        artifacts_path: str,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        max_length: int = 512,  # Menor para epitopos
        epochs: int = 10,
        save_interval: int = 2,
        base_model: str = "esmc_300m",
        mlm_probability: float = 0.15,
        warmup_steps: int = 1000,
        project: str = "esmc-pretraining",
        entity: str = None,
        run_name: str = None,
        seed: int = 42,
        resume_from_checkpoint: str = None,  # NOVO: caminho para checkpoint
        **kwargs
    ):
        # Configuração
        self.sequences = sequences
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_length = max_length
        self.epochs = epochs
        self.save_interval = save_interval
        self.base_model = base_model
        self.mlm_probability = mlm_probability
        self.warmup_steps = warmup_steps
        self.project = project
        self.entity = entity
        self.seed = seed
        self.global_step = 0
        
        # NOVO: Para continuar pré-treinamento
        self.resume_from_checkpoint = resume_from_checkpoint
        self.start_epoch = 1
        
        # Reprodutibilidade
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        time_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name if run_name is not None else f"pretrain_{base_model}_{time_stamp}"
        self.artifacts_path = Path(artifacts_path) / "pretraining" / self.run_name
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        # W&B
        wandb.init(
            project=project,
            entity=entity,
            name=self.run_name,
            config={
                "num_sequences": len(sequences),
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_length": max_length,
                "epochs": epochs,
                "base_model": base_model,
                "mlm_probability": mlm_probability,
                "warmup_steps": warmup_steps,
                "seed": seed,
            },
            reinit="finish_previous",
        )
        
        # Inicializar componentes
        self.tokenizer = get_tokenizer(self.base_model)
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        # NOVO: Carregar checkpoint se especificado
        self.load_checkpoint()
        
        print(f"🚀 Pré-treinador inicializado")
        print(f"   📊 {len(self.sequences)} sequências")
        print(f"   🧬 Modelo: {self.base_model}")
        print(f"   📱 Device: {self.device}")
        print(f"   💾 Artifacts: {self.artifacts_path}")
        
        if self.resume_from_checkpoint:
            print(f"   🔄 Continuando de: época {self.start_epoch}")
        else:
            print(f"   🆕 Novo treinamento")
    
    def setup_data(self):
        """Configurar dataset e dataloader"""
        # Filtrar sequências muito curtas (< 5 aminoácidos)
        filtered_sequences = [seq for seq in self.sequences if len(seq) >= 5]
        
        print(f"📋 Sequências filtradas: {len(filtered_sequences)}/{len(self.sequences)}")
        
        # Dataset
        self.dataset = MLMDataset(
            sequences=filtered_sequences,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            mlm_probability=self.mlm_probability
        )
        
        # DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"🔄 DataLoader: {len(self.dataloader)} batches")
    
    def setup_model(self):
        """Configurar modelo MLM"""
        self.model = ESMCForMLM(base_model=self.base_model).to(self.device)
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 Modelo carregado:")
        print(f"   📊 Total params: {total_params:,}")
        print(f"   🎯 Trainable params: {trainable_params:,}")
    
    def setup_optimizer(self):
        """Configurar otimizador e scheduler"""
        from transformers.optimization import get_cosine_schedule_with_warmup
        
        # Otimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        total_steps = len(self.dataloader) * self.epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"⚙️  Optimizer: AdamW (lr={self.lr}, wd={self.weight_decay})")
        print(f"📈 Scheduler: Cosine w/ warmup ({self.warmup_steps} steps)")
        print(f"🎯 Total steps: {total_steps}")
    
    def load_checkpoint(self):
        """Carregar checkpoint para continuar pré-treinamento"""
        if not self.resume_from_checkpoint:
            return
            
        if not os.path.exists(self.resume_from_checkpoint):
            print(f"❌ Checkpoint não encontrado: {self.resume_from_checkpoint}")
            return
            
        print(f"🔄 Carregando checkpoint: {self.resume_from_checkpoint}")
        
        checkpoint = torch.load(self.resume_from_checkpoint, map_location=self.device)
        
        # Carregar estado do modelo
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Carregar estado do otimizador
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Carregar estado do scheduler
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Carregar outras variáveis
        self.start_epoch = checkpoint['epoch'] + 1  # Próxima época
        self.global_step = checkpoint['global_step']
        
        print(f"✅ Checkpoint carregado!")
        print(f"   📅 Continuando da época {self.start_epoch}")
        print(f"   🎯 Global step: {self.global_step}")
        print(f"   📊 Loss anterior: {checkpoint['metrics']['avg_loss']:.4f}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Treinar uma época"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.dataloader,
            desc=f"Pré-treinamento",
            unit="batch"
        )
        
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
            
            # Forward pass
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log para W&B
            if self.global_step % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/step': self.global_step
                }, step=self.global_step)
        
        avg_loss = total_loss / num_batches
        return {'avg_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Salvar checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'metrics': metrics,
            'config': {
                'base_model': self.base_model,
                'max_length': self.max_length,
                'mlm_probability': self.mlm_probability,
            }
        }
        
        # Salvar checkpoint regular
        checkpoint_path = self.artifacts_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Salvar modelo pré-treinado (só os pesos do backbone)
        pretrained_path = self.artifacts_path / f"pretrained_esmc_epoch_{epoch}.pt"
        torch.save(self.model.esmc.state_dict(), pretrained_path)
        
        print(f"💾 Checkpoint salvo: {checkpoint_path.name}")
        print(f"🧬 Modelo pré-treinado salvo: {pretrained_path.name}")
    
    def run(self):
        """Executar pré-treinamento"""
        print(f"\n🚀 Iniciando pré-treinamento ({self.epochs} épocas)")
        print("="*50)
        
        # Verificar se há épocas para executar
        if self.start_epoch > self.epochs:
            print(f"⚠️  Aviso: Época inicial ({self.start_epoch}) > épocas totais ({self.epochs})")
            print(f"💡 Nada a executar. Use --pretrain-epochs {self.start_epoch + 5} para continuar")
            
            # Retornar o melhor modelo existente se disponível
            best_pretrained_path = self.artifacts_path / "best_pretrained_esmc.pt"
            if best_pretrained_path.exists():
                return str(best_pretrained_path)
            else:
                # Tentar encontrar o último modelo
                last_epoch = self.start_epoch - 1
                last_pretrained_path = self.artifacts_path / f"pretrained_esmc_epoch_{last_epoch}.pt"
                if last_pretrained_path.exists():
                    print(f"🎯 Usando modelo da época {last_epoch}")
                    return str(last_pretrained_path)
                else:
                    raise RuntimeError("Nenhum modelo pré-treinado encontrado para continuar")
        
        # Inicializar metrics para caso nenhuma época seja executada
        metrics = {'avg_loss': 0.0}
        epochs_executed = 0
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"\n📅 Época {epoch}/{self.epochs}")
            
            # Treinar
            metrics = self.train_epoch()
            epochs_executed += 1
            
            # Log época
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': metrics['avg_loss']
            }, step=self.global_step)
            
            print(f"   📊 Loss médio: {metrics['avg_loss']:.4f}")
            
            # Salvar checkpoint
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, metrics)
        
        # Salvar modelo final apenas se executou pelo menos uma época
        if epochs_executed > 0:
            print(f"\n✅ Pré-treinamento concluído!")
            self.save_checkpoint(self.epochs, metrics)
            
            # Salvar melhor modelo como "best"
            best_pretrained_path = self.artifacts_path / "best_pretrained_esmc.pt"
            final_pretrained_path = self.artifacts_path / f"pretrained_esmc_epoch_{self.epochs}.pt"
            
            import shutil
            shutil.copy2(final_pretrained_path, best_pretrained_path)
            print(f"🏆 Melhor modelo salvo: {best_pretrained_path.name}")
        else:
            print(f"\n✅ Nenhuma época executada (já completado)")
            best_pretrained_path = self.artifacts_path / "best_pretrained_esmc.pt"
        
        wandb.finish()
        
        return str(best_pretrained_path)


def load_sequences_from_files(pos_file: str, neg_file: str) -> List[str]:
    """Carregar sequências dos arquivos de dados"""
    sequences = []
    
    # Carregar positivos
    if os.path.exists(pos_file):
        with open(pos_file, 'r') as f:
            sequences.extend([line.strip() for line in f if line.strip()])
    
    # Carregar negativos
    if os.path.exists(neg_file):
        with open(neg_file, 'r') as f:
            sequences.extend([line.strip() for line in f if line.strip()])
    
    # Remover duplicatas
    sequences = list(set(sequences))
    
    print(f"📊 Carregadas {len(sequences)} sequências únicas")
    return sequences


def list_pretraining_checkpoints(dataset: str, model: str) -> List[Path]:
    """Listar checkpoints de pré-treinamento disponíveis"""
    base_path = Path(__file__).parent
    pretraining_dir = base_path / dataset / "pretraining"
    
    if not pretraining_dir.exists():
        print(f"❌ Pasta de pré-treinamento não encontrada: {pretraining_dir}")
        return []
    
    # Buscar por diretórios de pré-treinamento para o modelo
    pattern = f"pretrain_{model}_*"
    runs = list(pretraining_dir.glob(pattern))
    
    if not runs:
        print(f"❌ Nenhum pré-treinamento encontrado para {model}")
        return []
    
    # Ordenar por timestamp (mais recente primeiro)
    runs.sort(key=lambda x: x.name.split('_')[-1], reverse=True)
    
    print(f"\n📂 Pré-treinamentos disponíveis para {dataset}/{model}:")
    
    all_checkpoints = []
    for i, run_dir in enumerate(runs):
        print(f"\n   🗂️  {i+1}. {run_dir.name}")
        
        # Listar checkpoints neste run
        checkpoints = list(run_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        if checkpoints:
            print(f"      📁 Checkpoints disponíveis:")
            for cp in checkpoints:
                epoch = cp.stem.split('_')[-1]
                size_mb = cp.stat().st_size / (1024*1024)
                print(f"         • Época {epoch}: {cp.name} ({size_mb:.1f}MB)")
                all_checkpoints.append(cp)
        
        # Verificar se tem backbone pronto
        backbone_files = [
            run_dir / "best_pretrained_esmc.pt",
            run_dir / f"pretrained_esmc_epoch_{len(checkpoints)}.pt"
        ]
        
        available_backbones = [f for f in backbone_files if f.exists()]
        if available_backbones:
            print(f"      🧬 Backbones prontos:")
            for bb in available_backbones:
                size_mb = bb.stat().st_size / (1024*1024)
                print(f"         • {bb.name} ({size_mb:.1f}MB)")
    
    return all_checkpoints 