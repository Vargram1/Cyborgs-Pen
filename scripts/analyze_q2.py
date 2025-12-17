import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

# ================= 1. Configuration & Setup =================
CSV_FILE = "data/final_dataset_human_vs_ai.csv"

# Determine computation device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Computation Device: {DEVICE}")

# ================= 2. Data Loading & Restructuring =================
print("Loading dataset...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

# Reshape data to Long Format for unified analysis
# Structure: [id, prompt, story, group]
data_list = []

target_cols = [
    ('human_story', 'Human'),
    ('ai_gpt_mini', 'AI_GPT_Mini'),
    ('ai_gemini_flash', 'AI_Gemini_Flash'),
    ('ai_claude_haiku', 'AI_Claude_Haiku'),
    ('ai_claude_sonnet_sota', 'AI_Claude_Sonnet'),
    ('ai_gemini_pro_sota', 'AI_Gemini_Pro')
]

print("Restructuring data...")
for idx, row in df.iterrows():
    prompt = row['prompt']
    for col, group in target_cols:
        text = row.get(col)
        # Filter: text must be non-null and have sufficient length
        if pd.notna(text) and len(str(text)) > 50:
            data_list.append({
                'id': row['id'],
                'prompt': prompt,
                'story': str(text),
                'group': group
            })

df_long = pd.DataFrame(data_list)
print(f"Preprocessing complete. Valid samples: {len(df_long)}")

# ================= 3. Semantic Similarity Analysis =================
print("\n--- Phase 1: Semantic Prompt Adherence ---")

# Load SBERT model (all-MiniLM-L6-v2) for efficient embedding generation
print("Loading SBERT model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)

# A. Encode Prompts (Deduplicated for efficiency)
print("Encoding prompts...")
unique_prompts = df_long['prompt'].unique()
prompt_emb_dict = {p: embedder.encode(p, convert_to_tensor=True, show_progress_bar=False) for p in unique_prompts}

# B. Encode Stories
print("Encoding stories...")
story_embeddings = embedder.encode(df_long['story'].tolist(), show_progress_bar=True, device=DEVICE, convert_to_tensor=True)

# C. Compute Cosine Similarity
print("Calculating cosine similarity...")
similarities = []

for i, row in df_long.iterrows():
    p_emb = prompt_emb_dict[row['prompt']]
    s_emb = story_embeddings[i]
    
    # Compute similarity between prompt vector and story vector
    sim = cosine_similarity(p_emb.cpu().reshape(1, -1), s_emb.cpu().reshape(1, -1))[0][0]
    similarities.append(sim)

df_long['semantic_similarity'] = similarities

# D. Visualization
plt.figure(figsize=(12, 6))
# Filter out groups with small sample sizes if necessary
plot_df = df_long[df_long['group'] != 'AI_Gemini_Pro']

sns.boxplot(x='group', y='semantic_similarity', data=plot_df, palette="viridis", showfliers=False)
plt.title("Semantic Adherence: Similarity between Prompt and Story")
plt.ylabel("Cosine Similarity Score")
plt.xlabel("Model Group")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/q2_semantic_similarity.png")
print("Saved similarity plot: q2_semantic_similarity.png")

# ================= 4. Distinctive Keyword Analysis =================
print("\n--- Phase 2: Lexical Fingerprinting ---")

def get_top_unique_words(text_series, top_n=15):
    """
    Identifies high-frequency terms excluding standard stop words.
    """
    vec = CountVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 1))
    try:
        X = vec.fit_transform(text_series)
        sum_words = X.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return [w[0] for w in words_freq[:top_n]]
    except ValueError:
        return []

# Comparison: Human vs. SOTA AI
print("Top words (Human):")
print(get_top_unique_words(df_long[df_long['group'] == 'Human']['story']))

print("Top words (AI - Claude Sonnet):")
print(get_top_unique_words(df_long[df_long['group'] == 'AI_Claude_Sonnet']['story']))

# ================= 5. BERTopic Modeling =================
print("\n--- Phase 3: Thematic Clustering (BERTopic) ---")

docs = df_long['story'].tolist()
groups = df_long['group'].tolist()

# Initialize and fit BERTopic model
# min_topic_size=20 ensures robust clusters
topic_model = BERTopic(embedding_model=embedder, min_topic_size=20, verbose=True)

print("Fitting model...")
topics, probs = topic_model.fit_transform(docs, embeddings=story_embeddings.cpu().numpy())

# Export Topic Data
freq = topic_model.get_topic_info()
freq.to_csv("outputs/q2_topic_info.csv", index=False)
print("Topic info saved.")

# Generate Visualizations
fig = topic_model.visualize_topics()
fig.write_html("outputs/q2_topic_visualization.html")

# Generate 'Topics per Class' to compare Human vs AI thematic distribution
print("Calculating topics per class...")
topics_per_class = topic_model.topics_per_class(docs, classes=groups)
fig_per_class = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)
fig_per_class.write_html("outputs/q2_topics_per_group.html")

print("Analysis complete. All outputs saved.")