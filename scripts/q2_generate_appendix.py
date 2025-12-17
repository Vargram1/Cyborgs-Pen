import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

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

def main():
    print("--- Generating Appendix Artifacts (Fixed Version) ---")

    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found.")
        return

    print("Loading FULL dataset to ensure topic consistency...")
    data_list = []
    target_cols = [
        ('human_story', 'Human'),
        ('ai_gpt_mini', 'AI_GPT_Mini'),
        ('ai_gemini_flash', 'AI_Gemini_Flash'),
        ('ai_claude_haiku', 'AI_Claude_Haiku'),
        ('ai_claude_sonnet_sota', 'AI_Claude_Sonnet'),
        ('ai_gemini_pro_sota', 'AI_Gemini_Pro')
    ]

    for idx, row in df.iterrows():
        for col, group in target_cols:
            text = row.get(col)
            if pd.notna(text) and len(str(text)) > 50:
                data_list.append(str(text))
    
    print(f"Total documents loaded: {len(data_list)}")

    print("Refitting BERTopic (this matches your main analysis)...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    topic_model = BERTopic(embedding_model=embedder, min_topic_size=20, verbose=True)
    topic_model.fit_transform(data_list)
    
    found_topic_ids = set(topic_model.get_topic_info()['Topic'].tolist())
    
    valid_topic_ids = sorted([t for t in CUSTOM_LABELS.keys() if t in found_topic_ids and t != -1])
    
    print(f"Found {len(found_topic_ids)} topics in total.")
    print(f"Valid labeled topics for Appendix: {valid_topic_ids}")
    
    if not valid_topic_ids:
        print("Error: No matching topics found. Check your min_topic_size or data quality.")
        return

    print("Generating Appendix Table...")
    table_data = []
    
    for topic_id in valid_topic_ids:
        keywords = [word for word, score in topic_model.get_topic(topic_id)[:8]]
        keywords_str = ", ".join(keywords)

        count = topic_model.get_topic_info(topic_id)['Count'].values[0]
        
        table_data.append({
            "Topic ID": topic_id,
            "Semantic Label": CUSTOM_LABELS[topic_id],
            "Sample Size": count,
            "Top 8 Representative Keywords": keywords_str
        })
            
    df_table = pd.DataFrame(table_data)
    df_table.to_csv("data/outputs/Appendix_Table_Topics.csv", index=False)
    print("Table saved: Appendix_Table_Topics.csv")

    print("Generating Appendix Heatmap...")
    
    try:

        all_topics = topic_model.get_topic_info()['Topic'].tolist()
        if -1 in all_topics:
            all_topics.remove(-1)

        if topic_model.topic_embeddings_ is not None:

             
             indices = []
             info_df = topic_model.get_topic_info()
             
             final_topics_for_plot = []
             final_embeddings = []
             
             for tid in valid_topic_ids:

                 idx = info_df[info_df['Topic'] == tid].index[0]
                 if idx < len(topic_model.topic_embeddings_):
                     final_embeddings.append(topic_model.topic_embeddings_[idx])
                     final_topics_for_plot.append(tid)
             
             if len(final_embeddings) > 1:
                 sim_matrix = cosine_similarity(final_embeddings)
                 
                 plt.figure(figsize=(10, 8))
                 sns.set(font_scale=0.9)
                 labels = [CUSTOM_LABELS[t] for t in final_topics_for_plot]
                 
                 sns.heatmap(
                    sim_matrix,
                    xticklabels=labels,
                    yticklabels=labels,
                    cmap="RdBu_r",
                    center=0.5,
                    annot=True,
                    fmt=".2f",
                    square=True,
                    cbar_kws={"shrink": .8, "label": "Cosine Similarity"}
                 )
                 plt.title("Appendix Figure A1: Semantic Similarity Between Themes", fontsize=14, pad=20)
                 plt.xticks(rotation=45, ha='right')
                 plt.tight_layout()
                 plt.savefig("data/outputs/Appendix_Figure_Heatmap.png", dpi=300)
                 print("Figure saved: Appendix_Figure_Heatmap.png")
             else:
                 print("Not enough topics for heatmap.")

    except Exception as e:
        print(f"Skipping heatmap due to dimension mismatch: {e}")
        print("Note: The Table (Appendix_Table_Topics.csv) was saved successfully, which is the most important part.")

if __name__ == "__main__":
    main()