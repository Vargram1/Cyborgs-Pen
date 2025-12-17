import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW  # Using PyTorch's native optimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. Configuration =================
CSV_FILE = "data/final_dataset_human_vs_ai.csv"
MODEL_NAME = "distilbert-base-uncased" # Lightweight BERT variant for efficiency
MAX_LEN = 256       # Token sequence length
BATCH_SIZE = 16     # Adjusted for RTX 5070 Ti memory
EPOCHS = 3          # Standard fine-tuning duration
LEARNING_RATE = 2e-5

# Computation Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation Device: {device}")

# ================= 2. Dataset Preparation =================

class NarrativeDataset(Dataset):
    """
    Custom PyTorch Dataset for handling narrative texts.
    """
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_datasets(file_path):
    """
    Splits the dataset into two distinct groups:
    1. Training Corpus: Humans + Base AI Models (GPT-4o-mini, Flash, Haiku)
    2. SOTA Test Corpus: Humans (Holdout) + Flagship AI Models (Claude Sonnet, Gemini Pro)
    
    This structure tests the classifier's ability to generalize to unseen, advanced AI models.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        exit()

    # Containers for data
    train_texts, train_labels = [], []
    sota_texts, sota_labels = [], []

    # --- A. Human Data (Class 0) ---
    # Strategy: Use majority for training, reserve 300 for the SOTA control group
    human_stories = df['human_story'].dropna().tolist()
    human_train = human_stories[:-300]
    human_sota_control = human_stories[-300:]

    for t in human_train:
        if len(str(t)) > 50:
            train_texts.append(t)
            train_labels.append(0) # Label 0: Human

    for t in human_sota_control:
        if len(str(t)) > 50:
            sota_texts.append(t)
            sota_labels.append(0)

    # --- B. Base AI Data (Class 1) - Training Only ---
    # Models: GPT-4o-mini, Gemini Flash, Claude Haiku
    base_cols = ['ai_gpt_mini', 'ai_gemini_flash', 'ai_claude_haiku']
    for col in base_cols:
        if col in df.columns:
            stories = df[col].dropna().tolist()
            for t in stories:
                if len(str(t)) > 50:
                    train_texts.append(t)
                    train_labels.append(1) # Label 1: AI

    # --- C. SOTA AI Data (Class 1) - Testing Only ---
    # Models: Claude 3.7 Sonnet, Gemini 3 Pro
    # These are never seen during training (Zero-shot evaluation)
    sota_cols = ['ai_claude_sonnet_sota', 'ai_gemini_pro_sota']
    for col in sota_cols:
        if col in df.columns:
            stories = df[col].dropna().tolist()
            for t in stories:
                if len(str(t)) > 50:
                    sota_texts.append(t)
                    sota_labels.append(1)

    return (train_texts, train_labels), (sota_texts, sota_labels)

# ================= 3. Training & Evaluation Logic =================

def train_epoch(model, data_loader, optimizer):
    """
    Performs one epoch of training.
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    total_examples = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_examples += labels.shape[0]
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / total_examples, np.mean(losses)

def evaluate(model, data_loader):
    """
    Evaluates the model on a specific dataset.
    Returns accuracy metrics and confusion matrix data.
    """
    model = model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(actual_labels, predictions)
    p, r, f1, _ = precision_recall_fscore_support(actual_labels, predictions, average='binary')
    cm = confusion_matrix(actual_labels, predictions)
    
    return acc, p, r, f1, cm

# ================= 4. Main Execution Pipeline =================

def main():
    print("--- Starting Q3 Classification Experiment ---")
    
    # 1. Load and Split Data
    (train_txt, train_lbl), (sota_txt, sota_lbl) = prepare_datasets(CSV_FILE)
    print(f"Training Corpus Size: {len(train_txt)} (Base Models + Human)")
    print(f"SOTA Test Corpus Size: {len(sota_txt)} (Flagship Models + Human Control)")

    # Split Training into Train/Validation (85/15)
    tr_txt, val_txt, tr_lbl, val_lbl = train_test_split(
        train_txt, train_lbl, test_size=0.15, random_state=42
    )

    # 2. Tokenization
    print("Initializing Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_ds = NarrativeDataset(tr_txt, tr_lbl, tokenizer, MAX_LEN)
    val_ds = NarrativeDataset(val_txt, val_lbl, tokenizer, MAX_LEN)
    sota_ds = NarrativeDataset(sota_txt, sota_lbl, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    sota_loader = DataLoader(sota_ds, batch_size=BATCH_SIZE)

    # 3. Model Initialization
    print(f"Loading Pre-trained Model: {MODEL_NAME}...")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Fine-tuning Loop
    print(f"Starting fine-tuning for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        acc, loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | Train Acc: {acc:.4f}")
        
        # Validate on standard data
        val_acc, _, _, _, _ = evaluate(model, val_loader)
        print(f"  -> Validation Acc: {val_acc:.4f}")

    # 5. Final Evaluation & Reporting
    print("\n--- Experiment Results ---")
    
    # Evaluation A: Standard Validation (In-Distribution)
    v_acc, v_p, v_r, v_f1, v_cm = evaluate(model, val_loader)
    print(f"[Standard Validation] Accuracy: {v_acc:.4f} | F1: {v_f1:.4f}")

    # Evaluation B: SOTA Challenge (Out-of-Distribution)
    s_acc, s_p, s_r, s_f1, s_cm = evaluate(model, sota_loader)
    print(f"[SOTA Challenge Test] Accuracy: {s_acc:.4f} | F1: {s_f1:.4f}")
    
    # 6. Export Results
    # Save Metrics
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Base_Validation": [v_acc, v_p, v_r, v_f1],
        "SOTA_Challenge": [s_acc, s_p, s_r, s_f1]
    })
    metrics.to_csv("outputs/q3_classification_metrics.csv", index=False)
    print("Metrics saved to q3_classification_metrics.csv")

    # Save Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(v_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Human', 'Predicted AI'],
                yticklabels=['Actual Human', 'Actual AI'])
    axes[0].set_title("Standard Validation (Base Models)")
    
    sns.heatmap(s_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Predicted Human', 'Predicted AI'],
                yticklabels=['Actual Human', 'Actual AI'])
    axes[1].set_title("SOTA Challenge (Claude Sonnet / Gemini Pro)")
    
    plt.tight_layout()
    plt.savefig("outputs/q3_confusion_matrices.png")
    print("Visualizations saved to q3_confusion_matrices.png")
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()