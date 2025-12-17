import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from empath import Empath
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. Initialization =================

# Check and download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    print("Downloading required NLTK datasets...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')

# Initialize Analysis Tools
sia = SentimentIntensityAnalyzer() # VADER for sentiment polarity
lexicon = Empath() # Empath for semantic topic analysis (LIWC alternative)

# ================= 2. Feature Extraction Logic =================

def extract_features(text):
    """
    Extracts linguistic, syntactic, and semantic features from a text string.
    
    Args:
        text (str): The input story text.
        
    Returns:
        dict: A dictionary containing calculated metrics (TTR, Sentiment, etc.)
              or None if text is invalid.
    """
    if pd.isna(text) or text == "" or str(text).lower() == "nan":
        return None
    
    # Preprocessing: Tokenization
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    # Filter for alphanumeric tokens only to ensure accurate word counts
    words_lower = [w.lower() for w in words if w.isalnum()] 
    
    if len(words_lower) == 0: return None
    
    # --- A. Lexical Features ---
    # Type-Token Ratio (TTR): Indicates vocabulary richness
    ttr = len(set(words_lower)) / len(words_lower)
    
    # --- B. Syntactic Features ---
    # Average Sentence Length (ASL)
    avg_sent_len = len(words) / max(1, len(sentences))
    
    # Part-of-Speech (POS) Tagging
    pos_tags = nltk.pos_tag(words_lower)
    pos_counts = {'Noun': 0, 'Verb': 0, 'Adj': 0, 'Adv': 0}
    for word, tag in pos_tags:
        if tag.startswith('N'): pos_counts['Noun'] += 1
        elif tag.startswith('V'): pos_counts['Verb'] += 1
        elif tag.startswith('J'): pos_counts['Adj'] += 1
        elif tag.startswith('R'): pos_counts['Adv'] += 1
    
    # Normalize counts by total word count
    pos_ratios = {k: v / len(words_lower) for k, v in pos_counts.items()}
    
    # --- C. Sentiment Analysis (VADER) ---
    # Compound score: -1 (Negative) to +1 (Positive)
    vader_scores = sia.polarity_scores(text)
    
    # --- D. Emotional Analysis (NRC Lexicon) ---
    # Quantifies specific emotions: fear, anger, joy, sadness, etc.
    emotion_obj = NRCLex(text)
    raw_emotions = emotion_obj.raw_emotion_scores
    
    target_emotions = ['fear', 'anger', 'joy', 'sadness', 'surprise', 'trust']
    emotion_scores = {f"NRC_{k}": raw_emotions.get(k, 0) / len(words_lower) for k in target_emotions}

    # --- E. Psychometric Topics (Empath) ---
    # Selected categories relevant to creative narrative analysis
    target_topics = ['emotional', 'violence', 'family', 'science', 'death', 'negative_emotion', 'positive_emotion']
    empath_res = lexicon.analyze(text, categories=target_topics, normalize=True)
    
    # Aggregation
    features = {
        'Word_Count': len(words_lower),
        'TTR': ttr,
        'Avg_Sent_Len': avg_sent_len,
        'VADER_Compound': vader_scores['compound'],
        **pos_ratios,
        **emotion_scores,
        **empath_res
    }
    
    return features

# ================= 3. Main Execution Flow =================

def main():
    print("Starting Q1 Linguistic Feature Analysis...")
    
    # Load Dataset
    csv_file = "data/final_dataset_human_vs_ai.csv"
    try:
        df = pd.read_csv(csv_file)
        print(f"Data loaded: {len(df)} rows.")
    except FileNotFoundError:
        print("Error: Input CSV file not found.")
        return

    results_list = []
    
    # Map CSV columns to analytical group labels
    target_cols = [
        ('human_story', 'Human'),
        ('ai_gpt_mini', 'AI_GPT_Mini'),
        ('ai_gemini_flash', 'AI_Gemini_Flash'),
        ('ai_claude_haiku', 'AI_Claude_Haiku'),
        ('ai_claude_sonnet_sota', 'AI_Claude_Sonnet'),
        ('ai_gemini_pro_sota', 'AI_Gemini_Pro')
    ]

    print("Extracting features...")
    
    # Iterate through dataset
    for idx, row in df.iterrows():
        if idx % 100 == 0: 
            print(f"Processing row {idx}...")
        
        for col, group_label in target_cols:
            text = row.get(col)
            
            # Extract features only if text exists
            feats = extract_features(text)
            
            if feats:
                feats['id'] = row['id']
                feats['Group'] = group_label
                results_list.append(feats)

    # Export Results
    df_res = pd.DataFrame(results_list)
    output_file = "outputs/q1_features.csv"
    df_res.to_csv(output_file, index=False)
    print(f"Feature extraction complete. Saved to {output_file}")

# ================= 4. Visualization =================
    
    sns.set(style="whitegrid")

    plot_configs = [
        ('TTR', '(a) Vocabulary Richness'),
        ('Avg_Sent_Len', '(b) Sentence Complexity'),
        ('VADER_Compound', '(c) Sentiment Polarity'),
        ('NRC_joy', '(d) Joy Emotion'),
        ('violence', '(e) Violence Topic')
    ]
    
    plt.figure(figsize=(18, 10))
    
    for i, (metric, title) in enumerate(plot_configs):
        if metric not in df_res.columns: continue
        
        plt.subplot(2, 3, i+1)

        plot_data = df_res[df_res['Group'] != 'AI_Gemini_Pro'] 
        
        sns.boxplot(x='Group', y=metric, data=plot_data, palette="Set3", showfliers=False)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.xlabel("")
        plt.ylabel(metric)

    plt.tight_layout()
    plot_filename = "outputs/q1_combined_features.png"
    plt.savefig(plot_filename)
    print(f"Visualization saved to {plot_filename}")

if __name__ == "__main__":
    main()