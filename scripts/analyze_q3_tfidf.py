import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ================= 1. Configuration =================
CSV_FILE = "data/final_dataset_human_vs_ai.csv"
MAX_FEATURES = 5000  # Limit vocabulary size

# ================= 2. Data Preparation =================

def prepare_datasets(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: CSV file not found.")
        exit()

    train_texts, train_labels = [], []
    sota_texts, sota_labels = [], []

    # A. Human Data
    human_stories = df['human_story'].dropna().tolist()
    human_train = human_stories[:-300]
    human_sota_control = human_stories[-300:]

    for t in human_train:
        if len(str(t)) > 50:
            train_texts.append(str(t))
            train_labels.append(0)

    for t in human_sota_control:
        if len(str(t)) > 50:
            sota_texts.append(str(t))
            sota_labels.append(0)

    # B. Base AI (Training)
    base_cols = ['ai_gpt_mini', 'ai_gemini_flash', 'ai_claude_haiku']
    for col in base_cols:
        if col in df.columns:
            stories = df[col].dropna().tolist()
            for t in stories:
                if len(str(t)) > 50:
                    train_texts.append(str(t))
                    train_labels.append(1)

    # C. SOTA AI (Testing)
    sota_cols = ['ai_claude_sonnet_sota', 'ai_gemini_pro_sota']
    for col in sota_cols:
        if col in df.columns:
            stories = df[col].dropna().tolist()
            for t in stories:
                if len(str(t)) > 50:
                    sota_texts.append(str(t))
                    sota_labels.append(1)

    return (train_texts, train_labels), (sota_texts, sota_labels)

# ================= 3. Main Execution =================

def main():
    print("--- Starting TF-IDF + Logistic Regression Analysis ---")
    
    # 1. Load Data
    (train_txt, train_lbl), (sota_txt, sota_lbl) = prepare_datasets(CSV_FILE)
    print(f"Training Set: {len(train_txt)} samples")
    print(f"SOTA Test Set: {len(sota_txt)} samples")

    # Split Training
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        train_txt, train_lbl, test_size=0.2, random_state=42
    )

    # 2. Vectorization
    print(f"Vectorizing text (Max Features: {MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words='english')
    
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    X_sota = vectorizer.transform(sota_txt)

    # 3. Training
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)

    # 4. Evaluation Helper
    def get_metrics(name, X, y_true):
        y_pred = clf.predict(X)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n[{name} Results]")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall (Detection Rate): {r:.4f}")
        return [acc, p, r, f1], cm

    # 5. Run Evaluation
    val_metrics, val_cm = get_metrics("Standard Validation", X_val, y_val)
    
    sota_metrics, sota_cm = get_metrics("SOTA Challenge", X_sota, sota_lbl)

    # 6. Save Results
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Base_Validation": val_metrics,
        "SOTA_Challenge": sota_metrics
    })
    metrics_df.to_csv("data/outputs/q3_tfidf_metrics.csv", index=False)
    print("\nMetrics saved to data/outputs/q3_tfidf_metrics.csv")

    # 7. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Predicted Human', 'Predicted AI'],
                yticklabels=['Actual Human', 'Actual AI'])
    axes[0].set_title(f"Standard Validation (Base AI)\nAcc: {val_metrics[0]:.2%}")
    
    sns.heatmap(sota_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Predicted Human', 'Predicted AI'],
                yticklabels=['Actual Human', 'Actual AI'])
    axes[1].set_title(f"SOTA Challenge (Claude/Gemini)\nAcc: {sota_metrics[0]:.2%}")
    
    plt.tight_layout()
    plt.savefig("data/outputs/q3_tfidf_confusion_matrices.png")
    print("Confusion matrices saved to data/outputs/q3_tfidf_confusion_matrices.png")

    # Calculate Gap
    gap = val_metrics[2] - sota_metrics[2]
    print(f"\n*** RESULT HIGHLIGHT ***")
    print(f"Recall Gap: {(gap*100):.2f}%")

if __name__ == "__main__":
    main()