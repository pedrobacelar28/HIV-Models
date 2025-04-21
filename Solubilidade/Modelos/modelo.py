import torch
from transformers import EsmForSequenceClassification, EsmTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
from torch.optim import AdamW
import numpy as np
import os
from tqdm import tqdm
import random
import json
import datetime


seed = 42

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(seed)



# Classe para carregar e preparar o dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=850):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        # Tokenize sequence
        inputs = self.tokenizer(sequence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)  # Extrair a máscara de atenção
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Função para carregar arquivos com uma sequência de proteína por linha
def load_protein_sequences(filename, label):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    sequences = []
    labels = []

    # Para cada linha do arquivo, trata-se como uma sequência
    for line in lines:
        sequence = line.strip()
        if sequence:  # Ignorar linhas vazias
            sequences.append(sequence)
            labels.append(label)

    return sequences, labels

# Carregar dados dos arquivos onde cada linha é uma sequência de proteína
allergic_sequences, allergic_labels = load_protein_sequences('simB.txt', 1)
non_allergic_sequences, non_allergic_labels = load_protein_sequences('naoB.txt', 0)

# Combinar dados
sequences = allergic_sequences + non_allergic_sequences
labels = allergic_labels + non_allergic_labels

# Inicializar tokenizer e modelo ESM-3
from transformers import EsmTokenizer, EsmForSequenceClassification

# Substituindo por um modelo mais potente, por exemplo, o modelo ESM2-t36_3B
tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')
model = EsmForSequenceClassification.from_pretrained('facebook/esm2_t12_35M_UR50D', num_labels=2)


# Preparar dataset
from sklearn.model_selection import train_test_split

# After creating your dataset
dataset = ProteinDataset(sequences, labels, tokenizer)

# Get all the labels to use for stratification
all_labels = [labels[i] for i in range(len(dataset))]

# Calculate indices for stratified train/val/test split
train_idx, temp_idx = train_test_split(
    range(len(dataset)), 
    test_size=0.1,  # 10% for val+test
    stratify=all_labels,
    random_state=42
)

# Split the remaining 10% into validation and test sets
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,  # 5% for test, 5% for validation
    stratify=[all_labels[i] for i in temp_idx],
    random_state=42
)

# Create subset samplers
from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)

# Create data loaders with samplers
train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=8, sampler=val_sampler)
test_loader = DataLoader(dataset, batch_size=8, sampler=test_sampler)

# Print class distribution in each split to verify
def check_distribution(indices, all_labels):
    split_labels = [all_labels[i] for i in indices]
    class_0 = split_labels.count(0)
    class_1 = split_labels.count(1)
    total = len(split_labels)
    print(f"Class 0: {class_0} ({class_0/total:.2%}), Class 1: {class_1} ({class_1/total:.2%})")

print("Original distribution:")
class_0 = all_labels.count(0)
class_1 = all_labels.count(1)
print(f"Class 0: {class_0} ({class_0/len(all_labels):.2%}), Class 1: {class_1} ({class_1/len(all_labels):.2%})")

print("Train distribution:")
check_distribution(train_idx, all_labels)

print("Validation distribution:")
check_distribution(val_idx, all_labels)

print("Test distribution:")
check_distribution(test_idx, all_labels)

# Configurar otimizador e scheduler
optimizer = AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
total_steps = len(train_loader) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.02 * total_steps), num_training_steps=total_steps)

# Função de treinamento
def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    for input_ids, attention_mask, labels in tqdm(loader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)  # Mover a máscara para o device
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)  # Passar a máscara para o modelo
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    accuracy = correct / total
    average_loss = total_loss / len(loader)
    return accuracy, average_loss

def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Validation"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)  # Mover a máscara para o device
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)  # Passar a máscara para o modelo
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    average_loss = total_loss / len(loader)
    return accuracy, average_loss

# Função para calcular as métricas no conjunto de teste
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score

# Função para calcular as métricas no conjunto de teste, incluindo AUC
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Testing"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)  # Mover a máscara para o device
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)  # Passar a máscara para o modelo
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            predictions = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)  # Converter para numpy
    
     # Replace this section with the following:
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Use zero_division=0 parameter to handle cases with no samples in a class
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    # Force a 2x2 confusion matrix even if only one class is present
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Add protection against division by zero
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision  # Positive Predictive Value is the same as precision
    
    # Check if we have both classes before calculating AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        print("Warning: Cannot calculate AUC because only one class is present in true labels")
        auc = 0.5  # Default value for random classifier
    
    return accuracy, precision, recall, specificity, ppv, f1, mcc, auc

# Função para atualizar e salvar métricas em arquivo JSON único
def update_and_save_metrics(metrics_file, new_data):
    # Verificar se o arquivo já existe
    if os.path.exists(metrics_file):
        # Carregar dados existentes
        with open(metrics_file, 'r') as f:
            data = json.load(f)
    else:
        # Criar estrutura inicial
        data = {"epochs": [], "test": None}
    
    # Adicionar novos dados (seja uma nova época ou resultados de teste)
    if "epoch" in new_data:  # Se for dados de uma época
        data["epochs"].append(new_data)
    else:  # Se for dados de teste
        data["test"] = new_data
    
    # Salvar dados atualizados
    with open(metrics_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Metrics updated and saved to {metrics_file}")

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Criar pasta principal 'model' se não existir
os.makedirs('model', exist_ok=True)

# Criar uma pasta com timestamp para o run atual
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join('model', f'run_{timestamp}')
os.makedirs(run_dir, exist_ok=True)

# Caminhos para salvar o modelo, tokenizer e métricas
model_dir = os.path.join(run_dir, 'model')
tokenizer_dir = os.path.join(run_dir, 'tokenizer')
metrics_file = os.path.join(run_dir, 'model_metrics.json')

print(f"Iniciando treinamento, resultados serão salvos em: {run_dir}")

# Loop de treinamento e validação
epochs = 5
for epoch in range(epochs):
    epoch_num = epoch + 1
    
    # Treinar o modelo
    train_accuracy, train_loss = train(model, train_loader, optimizer, scheduler, device)
    
    # Validar o modelo e calcular métricas detalhadas
    val_accuracy, val_loss = validate(model, val_loader, device)
    val_accuracy_detailed, val_precision, val_recall, val_specificity, val_ppv, val_f1, val_mcc, val_auc = evaluate(model, val_loader, device)
    
    # Imprimir resultados
    print(f"Epoch {epoch_num}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    print(f"Val Accuracy (detailed): {val_accuracy_detailed:.4f}")
    print(f"Val Precision (PPV): {val_precision:.4f}")
    print(f"Val Recall (Sensitivity): {val_recall:.4f}")
    print(f"Val Specificity: {val_specificity:.4f}")
    print(f"Val F1 Score: {val_f1:.4f}")
    print(f"Val MCC: {val_mcc:.4f}")
    print(f"Val AUC: {val_auc:.4f}")
    
    # Criar um dicionário com as métricas da época atual
    epoch_metrics = {
        "epoch": epoch_num,
        "training": {
            "loss": train_loss,
            "accuracy": train_accuracy
        },
        "validation": {
            "loss": val_loss,
            "accuracy": val_accuracy,
            "accuracy_detailed": val_accuracy_detailed,
            "precision": val_precision,
            "recall": val_recall,
            "specificity": val_specificity,
            "ppv": val_ppv,
            "f1": val_f1,
            "mcc": val_mcc,
            "auc": val_auc
        }
    }
    
    # Atualizar e salvar métricas no arquivo JSON único
    update_and_save_metrics(metrics_file, epoch_metrics)

# Avaliação final no conjunto de teste
test_accuracy, test_precision, test_recall, test_specificity, test_ppv, test_f1, test_mcc, test_auc = evaluate(model, test_loader, device)

# Imprimir resultados do teste
print("Final Test Results:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision (PPV): {test_precision:.4f}")
print(f"Test Recall (Sensitivity): {test_recall:.4f}")
print(f"Test Specificity: {test_specificity:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test MCC: {test_mcc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Criar um dicionário com os resultados do teste
test_metrics = {
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "specificity": test_specificity,
    "ppv": test_ppv,
    "f1": test_f1,
    "mcc": test_mcc,
    "auc": test_auc
}

# Atualizar o arquivo JSON com os resultados do teste
update_and_save_metrics(metrics_file, test_metrics)

# Salvar o modelo e tokenizer na pasta específica deste run
model.save_pretrained(model_dir)
tokenizer.save_pretrained(tokenizer_dir)

# Salvar um arquivo README com informações sobre o run
run_info = {
    "seed" : seed,
    "test_results": test_metrics
}

with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
    json.dump(run_info, f, indent=4)

print(f"Modelo, tokenizer e métricas salvos com sucesso em {run_dir}")
