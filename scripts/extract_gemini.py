import pandas as pd
df = pd.read_csv("data/final_dataset_human_vs_ai.csv")

gemini_subset = df[df['ai_gemini_pro_sota'].notna()]

result = gemini_subset[['id', 'prompt', 'ai_gemini_pro_sota']]

print(f"--- Extraction Report ---")
print(f"Total rows in original file: {len(df)}")
print(f"Successfully extracted Gemini 3 Pro stories: {len(result)}")

output_file = "data/gemini_pro_only.csv"
result.to_csv(output_file, index=False)

print(f"Saved to: {output_file}")