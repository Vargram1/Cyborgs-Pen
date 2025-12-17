import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch

# ================= 1. Configuration =================
CSV_FILE = "data/final_dataset_human_vs_ai.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CUSTOM_LABELS = {
    -1: "Outliers",
    0: "Interpersonal Drama (Realism)",
    1: "High Fantasy (Epic)", 
    2: "AI Artifact (The 'Elara' Trope)", 
    3: "Cyberpunk / Tech", 
    4: "War / Conflict", 
    5: "Existential / Mortality", 
    6: "Romance / Character Focus",
    7: "Superpowers / Abilities",
    8: "Sci-Fi / Space Opera", 
    9: "Meta-Fiction / Breaking 4th Wall"
}

# ================= 2. AGGREGATION MAPPING =================
GROUP_MAPPING = {
    'human_story':           'Human',
    
    # Base Models Group
    'ai_gpt_mini':           'Base AI',
    'ai_gemini_flash':       'Base AI',
    'ai_claude_haiku':       'Base AI',
    
    # SOTA Models Group
    'ai_claude_sonnet_sota': 'SOTA AI',
    'ai_gemini_pro_sota':    'SOTA AI'
}

# (Base -> SOTA -> Human)
DISPLAY_ORDER = ['Base AI', 'SOTA AI', 'Human']

PALETTE = {
    'Base AI': '#6BAED6',
    'SOTA AI': '#2CA02C',
    'Human':   '#E377C2'
}

def main():
    print("--- Starting Aggregated Q2 Visualization ---")
    print(f"Target Groups: {DISPLAY_ORDER}")

    # 1. Load Data & Apply Aggregation
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Error: CSV not found!")
        return
    
    docs = []
    classes = []

    for idx, row in df.iterrows():
        for col_name, super_group in GROUP_MAPPING.items():
            text = row.get(col_name)
            if pd.notna(text) and len(str(text)) > 50:
                docs.append(str(text))
                classes.append(super_group)
    
    print(f"Data Aggregated: {len(docs)} total documents processed.")
    print(f"Group Sizes: {pd.Series(classes).value_counts().to_dict()}")

    # 2. Fit BERTopic
    print("Fitting BERTopic on aggregated data...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    topic_model = BERTopic(embedding_model=embedder, min_topic_size=20, verbose=True)
    topic_model.fit_transform(docs)

    # 3. Calculate Topics per Class (Super Group)
    print("Calculating statistics for super groups...")
    topics_per_class = topic_model.topics_per_class(docs, classes=classes)
    
    # Map Labels
    topics_per_class['Topic_Label'] = topics_per_class['Topic'].map(CUSTOM_LABELS)
    
    # Filter Valid Topics
    plot_data = topics_per_class[topics_per_class['Topic'].isin(CUSTOM_LABELS.keys())].copy()
    plot_data = plot_data[plot_data['Topic'] != -1]

    # 4. Normalization (Group-Level)
    group_counts = pd.Series(classes).value_counts().to_dict()
    
    plot_data['Percentage'] = plot_data.apply(
        lambda row: (row['Frequency'] / group_counts[row['Class']]) * 100, 
        axis=1
    )

    # 5. Plotting
    print("Generating Aggregated Figure 3...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 8))

    ax = sns.barplot(
        data=plot_data,
        x='Topic_Label',
        y='Percentage',
        hue='Class',
        hue_order=DISPLAY_ORDER, 
        palette=PALETTE,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9
    )

    # Formatting
    plt.title("Figure 3: Thematic Distribution by Generation (Aggregated)", fontsize=16, weight='bold', pad=20)
    plt.xlabel("Narrative Theme (Topic)", fontsize=14, labelpad=10)
    plt.ylabel("Prevalence within Group (%)", fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=11)
    
    # Update Legend
    plt.legend(title='Author Group', title_fontsize='12', loc='upper right')

    plt.tight_layout()
    
    output_png = "data/outputs/Figure3_Base_SOTA_Human.png"
    plt.savefig(output_png, dpi=300)
    print(f"Aggregated chart saved to: {output_png}")

if __name__ == "__main__":
    main()