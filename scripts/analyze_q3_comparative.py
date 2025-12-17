import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================= 1. Configuration =================
CSV_FILE = "data/final_dataset_human_vs_ai.csv"
OUTPUT_PREDICTIONS = "data/outputs/q3_comparative_predictions.csv"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Computation Device: {DEVICE}")

# ================= 2. Data Preparation =================

class NarrativeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
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
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def load_data(file_path):
    """
    Loads data and creates strict Training (Base) vs Testing (SOTA) splits
    while preserving IDs for later feature linking.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        exit()

    # Containers
    train_data = []
    test_data = []

    # A. Human Data
    human_subset = df[df['human_story'].notna()]
    # Last 300 for SOTA test
    human_train = human_subset.iloc[:-300]
    human_test = human_subset.iloc[-300:]

    for _, row in human_train.iterrows():
        if len(str(row['human_story'])) > 50:
            train_data.append({'text': row['human_story'], 'label': 0})
    
    for _, row in human_test.iterrows():
        if len(str(row['human_story'])) > 50:
            test_data.append({'id': row['id'], 'text': row['human_story'], 'label': 0, 'source': 'Human'})

    # B. Base AI (Training Only)
    for col in ['ai_gpt_mini', 'ai_gemini_flash', 'ai_claude_haiku']:
        subset = df[df[col].notna()]
        for _, row in subset.iterrows():
            if len(str(row[col])) > 50:
                train_data.append({'text': row[col], 'label': 1})

    # C. SOTA AI (Testing Only)
    for col in ['ai_claude_sonnet_sota', 'ai_gemini_pro_sota']:
        subset = df[df[col].notna()]
        for _, row in subset.iterrows():
            if len(str(row[col])) > 50:
                test_data.append({'id': row['id'], 'text': row[col], 'label': 1, 'source': 'SOTA_AI'})

    return pd.DataFrame(train_data), pd.DataFrame(test_data)

# ================= 3. Main Analysis =================

def main():
    print("Starting Comparative Analysis (TF-IDF vs. BERT)...")
    
    # 1. Load Data
    df_train, df_test = load_data(CSV_FILE)
    print(f"Training Samples: {len(df_train)}")
    print(f"SOTA Test Samples: {len(df_test)}")

    # ---------------- PART A: TF-IDF Baseline ----------------
    print("\n[Method 1] Running TF-IDF + Logistic Regression...")
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vec.fit_transform(df_train['text'])
    X_test = vec.transform(df_test['text'])
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, df_train['label'])
    
    # Generate predictions
    tfidf_preds = clf.predict(X_test)
    print(f"TF-IDF Accuracy on SOTA: {accuracy_score(df_test['label'], tfidf_preds):.4f}")

    # ---------------- PART B: DistilBERT ----------------
    print("\n[Method 2] Running DistilBERT (Fine-tuning)...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    
    # Prepare DataLoader
    train_ds = NarrativeDataset(df_train['text'].values, df_train['label'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Fast Training (1 Epoch is sufficient to demonstrate the gap)
    print("Training BERT (1 Epoch)...")
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Inference on Test Set
    print("Generating BERT predictions...")
    test_ds = NarrativeDataset(df_test['text'].values, df_test['label'].values, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    bert_preds = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.max(outputs.logits, dim=1)[1]
            bert_preds.extend(preds.cpu().tolist())

    print(f"BERT Accuracy on SOTA: {accuracy_score(df_test['label'], bert_preds):.4f}")

    # ---------------- PART C: Consolidate Results ----------------
    print("\nConsolidating results...")
    df_test['pred_tfidf'] = tfidf_preds
    df_test['pred_bert'] = bert_preds
    
    # Categorize outcomes for analysis
    # "Hidden AI" = AI samples where TF-IDF failed (0) but BERT succeeded (1)
    df_test['classification_category'] = 'Other'
    
    # Logic for AI samples
    ai_mask = df_test['label'] == 1
    
    # Case 1: The "Hidden" AI (Requires deep semantic detection)
    mask_hidden = ai_mask & (df_test['pred_tfidf'] == 0) & (df_test['pred_bert'] == 1)
    df_test.loc[mask_hidden, 'classification_category'] = 'Hidden_AI'
    
    # Case 2: The "Obvious" AI (Caught by both)
    mask_obvious = ai_mask & (df_test['pred_tfidf'] == 1) & (df_test['pred_bert'] == 1)
    df_test.loc[mask_obvious, 'classification_category'] = 'Obvious_AI'
    
    # Case 3: Human Baseline
    df_test.loc[df_test['label'] == 0, 'classification_category'] = 'Human'

    print(f"Identified {sum(mask_hidden)} 'Hidden AI' samples (Gap attribution targets).")
    
    df_test.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Prediction dataset saved to: {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    main()