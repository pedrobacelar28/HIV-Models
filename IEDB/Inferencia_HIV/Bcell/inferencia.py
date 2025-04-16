import torch
from transformers import EsmForSequenceClassification, EsmTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, roc_auc_score
import numpy as np
from tqdm import tqdm
import random
import os
import json
import datetime

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=30):
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
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

# Function to load protein sequences from file
def load_protein_sequences(filename, label):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    sequences = []
    labels = []

    for line in lines:
        sequence = line.strip()
        if sequence:  # Skip empty lines
            sequences.append(sequence)
            labels.append(label)

    return sequences, labels

# Function to evaluate the model
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Testing"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            predictions = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Handle cases with no samples in a class
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    
    # Force a 2x2 confusion matrix even if only one class is present
    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Protection against division by zero
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision  # Positive Predictive Value is the same as precision
    
    # Check if we have both classes before calculating AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        print("Warning: Cannot calculate AUC because only one class is present in true labels")
        auc = 0.5  # Default value for random classifier
    
    return accuracy, precision, recall, specificity, ppv, f1, mcc, auc, cm

def save_inference_results(model_path, metrics, conf_matrix):
    # Extract run ID from model path
    run_id = model_path.split('/')[-2]  # Assuming the format is '.../run_YYYYMMDD_HHMMSS/model'
    
    # Create results directory if it doesn't exist
    results_dir = './Resultados'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results filename with run ID
    results_file = os.path.join(results_dir, f"inference_{run_id}.json")
    
    # Prepare results dictionary
    results = {
        "model_path": model_path,
        "inference_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "accuracy": metrics[0],
            "precision": metrics[1],
            "recall": metrics[2],
            "specificity": metrics[3],
            "ppv": metrics[4],
            "f1": metrics[5],
            "mcc": metrics[6],
            "auc": metrics[7],
        },
        "confusion_matrix": {
            "array": conf_matrix.tolist(),
            "true_negatives": int(conf_matrix[0, 0]),
            "false_positives": int(conf_matrix[0, 1]),
            "false_negatives": int(conf_matrix[1, 0]),
            "true_positives": int(conf_matrix[1, 1])
        }
    }
    
    # Save to file
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results_file

def main():
    print("Loading model and tokenizer...")
    
    # Load the saved model and tokenizer
    model_path = '/scratch/pedro.bacelar/HIV-Models/IEDB/Modelos/MHC1/model/run_20250416_152440/model'  # Path where the model was saved
    tokenizer_path = '/scratch/pedro.bacelar/HIV-Models/IEDB/Modelos/MHC1/model/run_20250416_152440/tokenizer'  # Path where the tokenizer was saved
    
    try:
        tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
        model = EsmForSequenceClassification.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return
    
    # Load data for inference
    print("Loading data...")
    try:
        allergic_sequences, allergic_labels = load_protein_sequences('simBHIV.txt', 1)
        non_allergic_sequences, non_allergic_labels = load_protein_sequences('naoMHC1HIV.txt', 0)
        
        # Combine data
        sequences = allergic_sequences + non_allergic_sequences
        labels = allergic_labels + non_allergic_labels
        
        print(f"Loaded {len(allergic_sequences)} positive samples and {len(non_allergic_sequences)} negative samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare dataset for inference
    dataset = ProteinDataset(sequences, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Perform evaluation
    print("Running inference and calculating metrics...")
    accuracy, precision, recall, specificity, ppv, f1, mcc, auc, conf_matrix = evaluate(model, dataloader, device)
    
    # Pack metrics for saving
    metrics = (accuracy, precision, recall, specificity, ppv, f1, mcc, auc)
    
    # Save results to file
    results_file = save_inference_results(model_path, metrics, conf_matrix)
    print(f"\nResults saved to: {results_file}")
    
    # Print metrics
    print("\n===== EVALUATION METRICS =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (PPV): {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # Print confusion matrix
    print("\n===== CONFUSION MATRIX =====")
    print("Format: [[TN, FP], [FN, TP]]")
    print(conf_matrix)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"\nTrue Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

if __name__ == "__main__":
    main()