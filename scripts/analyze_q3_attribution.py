import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 1. Configuration =================
PREDICTIONS_FILE = "data/outputs/q3_comparative_predictions.csv"
FEATURES_FILE = "data/outputs/q1_features.csv"
OUTPUT_PLOT = "data/outputs/evidence_quantified_gap.png"

# ================= 2. Analysis Logic =================

def main():
    print("Starting Q3 Attribution Analysis...")
    
    try:
        preds_df = pd.read_csv(PREDICTIONS_FILE)
        feats_df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print("Error: Input files not found. Run previous scripts first.")
        return

    # We need to merge features (VADER, TTR) into the prediction results.
    # The merge logic relies on 'id'. 
    # Note: 'q1_features.csv' has multiple rows per ID (Human, GPT, Claude, etc).
    # We must filter Q1 to match the 'source' type in predictions.
    
    print("Linking Classification Results to Linguistic Features...")
    
    # Prepare Q1 data for merging: Filter for SOTA AI and Human rows only
    # (Because our predictions file only contains SOTA Test data)
    target_groups = ['Human', 'AI_Claude_Sonnet', 'AI_Gemini_Pro']
    feats_subset = feats_df[feats_df['Group'].isin(target_groups)].copy()
    
    # Merge strategy:
    # Since linking by ID alone is ambiguous (1 ID = multiple stories), we assume
    # the predictions file structure. A simpler approach for analysis is to 
    # extract features for the specific IDs identified in the categories.
    
    # Extract IDs for each category
    ids_hidden = preds_df[preds_df['classification_category'] == 'Hidden_AI']['id'].unique()
    ids_obvious = preds_df[preds_df['classification_category'] == 'Obvious_AI']['id'].unique()
    ids_human = preds_df[preds_df['classification_category'] == 'Human']['id'].unique()
    
    # Create comparison dataset
    comparison_data = []
    
    # Helper to fetch mean features for a list of IDs and a specific Group type
    def add_data(id_list, group_filter, label):
        # Select rows in features file that match ID list AND Group type
        # For AI categories, we look at SOTA groups. For Human, Human group.
        if label == 'Human':
            subset = feats_df[(feats_df['id'].isin(id_list)) & (feats_df['Group'] == 'Human')]
        else:
            subset = feats_df[(feats_df['id'].isin(id_list)) & (feats_df['Group'].isin(['AI_Claude_Sonnet', 'AI_Gemini_Pro']))]
        
        for _, row in subset.iterrows():
            comparison_data.append({
                'Category': label,
                'TTR': row['TTR'],
                'VADER_Compound': row['VADER_Compound']
            })

    print("Aggregating statistics...")
    add_data(ids_human, 'Human', 'Human')
    add_data(ids_hidden, 'AI', 'Hidden AI')
    add_data(ids_obvious, 'AI', 'Obvious AI')
    
    final_df = pd.DataFrame(comparison_data)
    
    # Calculate Means for the Report
    summary = final_df.groupby('Category')[['TTR', 'VADER_Compound']].mean()
    print("\n=== EVIDENCE TABLE (Attribution Results) ===")
    print(summary)
    print("------------------------------------------------")
    print("Interpretation:")
    print("1. Hidden AI should have HIGH TTR (mimicking Human vocabulary).")
    print("2. Hidden AI should have HIGH VADER (retaining AI emotional positivity).")
    
    # Visualization
    print("\nGenerating evidence visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Lexical Diversity (Why TF-IDF failed)
    sns.boxplot(x='Category', y='TTR', data=final_df, palette="Set2", ax=axes[0], showfliers=False)
    axes[0].set_title("Lexical Diversity (TTR)\n(Why TF-IDF failed on Hidden AI)")
    axes[0].set_ylabel("Type-Token Ratio")
    
    # Plot 2: Emotional Sentiment (Why BERT succeeded)
    sns.boxplot(x='Category', y='VADER_Compound', data=final_df, palette="Set2", ax=axes[1], showfliers=False)
    axes[1].set_title("Emotional Sentiment (VADER)\n(Why BERT detected Hidden AI)")
    axes[1].set_ylabel("Sentiment Score (1.0 = Positive)")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Visualization saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()