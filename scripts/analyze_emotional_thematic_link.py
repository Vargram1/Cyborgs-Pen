import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ================= 1. Configuration =================
INPUT_FILE = "outputs/q1_features.csv"
OUTPUT_PLOT = "outputs/evidence_q2_sentiment_link.png"

# ================= 2. Analysis Logic =================

def main():
    print("Starting Emotional-Thematic Link Analysis...")
    
    # Load Feature Data
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Data loaded: {len(df)} samples.")
    except FileNotFoundError:
        print("Error: 'q1_features.csv' not found. Please run Q1 analysis first.")
        return

    # --- Analysis A: Emotional Volatility (Variance) ---
    # Hypothesis: Humans engage in themes with higher emotional stakes (high variance),
    # while AI prefers 'safe' themes (low variance).
    print("\nCalculating Emotional Volatility (Standard Deviation of VADER scores)...")
    
    sentiment_variance = df.groupby('Group')['VADER_Compound'].std().sort_values(ascending=False)
    print("\n[Table 1] Emotional Volatility by Group:")
    print(sentiment_variance)
    
    # --- Analysis B: Negative Emotion Density ---
    # Hypothesis: AI thematic avoidance specifically targets conflict/negativity.
    # We use 'negative_emotion' (Empath) and 'NRC_fear' as proxies for conflict.
    print("\nCalculating Negative Emotion Density...")
    
    cols_to_analyze = ['negative_emotion', 'NRC_fear', 'NRC_sadness']
    neg_stats = df.groupby('Group')[cols_to_analyze].mean()
    
    print("\n[Table 2] Mean Negative Emotion Scores:")
    print(neg_stats)

    # ================= 3. Visualization =================
    print(f"\nGenerating evidence visualization ({OUTPUT_PLOT})...")
    
    # Prepare data for plotting (Focus on Negative Emotion as the strongest signal)
    # Reset index to make 'Group' a column for Seaborn
    plot_data = df.groupby('Group')['negative_emotion'].mean().reset_index()
    plot_data = plot_data.sort_values('negative_emotion', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Group', y='negative_emotion', data=plot_data, palette="magma")
    
    plt.title("Thematic Consequence: Density of Negative Emotional Content", fontsize=14)
    plt.ylabel("Mean Negative Emotion Score (Empath)", fontsize=12)
    plt.xlabel("Authorship Group", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Analysis complete. Visualization saved.")

if __name__ == "__main__":
    main()