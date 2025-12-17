import pandas as pd
import os

SOURCE_FILE = os.path.join("data", "valid.wp_source", "valid.wp_source")
TARGET_FILE = os.path.join("data", "valid.wp_target", "valid.wp_target")
OUTPUT_CSV = "data/writingprompts.csv"


print(f"1. Attempting to read prompts from: {SOURCE_FILE} ...")
try:
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines()]
    print(f"   -> Successfully loaded {len(prompts)} prompts.")
except FileNotFoundError:
    print(f"ERROR: Could not find file at {SOURCE_FILE}. Please check the folder structure.")
    exit()

print(f"2. Attempting to read stories from: {TARGET_FILE} ...")
try:
    with open(TARGET_FILE, 'r', encoding='utf-8') as f:
        stories = [line.strip() for line in f.readlines()]
    print(f"   -> Successfully loaded {len(stories)} stories.")
except FileNotFoundError:
    print(f"ERROR: Could not find file at {TARGET_FILE}.")
    exit()

min_len = min(len(prompts), len(stories))
prompts = prompts[:min_len]
stories = stories[:min_len]

print(f"3. Merging and cleaning data (Total: {min_len} rows)...")

df = pd.DataFrame({
    'prompt': prompts,
    'story': stories
})

df['story'] = df['story'].str.replace('<newline>', ' ')
df['prompt'] = df['prompt'].str.replace('<newline>', ' ')

df = df[df['story'].str.len() > 150]

print(f"4. Saving to CSV... (Valid rows after filtering: {len(df)})")
df.to_csv(OUTPUT_CSV, index=False)

print(f"Success! File generated: {OUTPUT_CSV}")