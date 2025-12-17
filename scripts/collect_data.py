import pandas as pd
from openai import OpenAI
import requests
import time
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from keys import OPENROUTER_API_KEY

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

INPUT_CSV_PATH = "data/writingprompts.csv"

OUTPUT_CSV_PATH = "data/final_dataset_human_vs_ai.csv"

TOTAL_SAMPLE_SIZE = 2000
SOTA_SAMPLE_SIZE = 300

MAX_WORKERS = 5

MODELS = {

    "gpt_mini": "openai/gpt-4o-mini",
    "gemini_flash": "google/gemini-2.0-flash-001",
    "claude_haiku": "anthropic/claude-3-haiku",

    "gemini_pro_sota": "google/gemini-3-pro-preview",
    "claude_sonnet_sota": "anthropic/claude-sonnet-4.5"
}


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)
csv_lock = threading.Lock()

def generate_story(prompt_text, model_id):
    """
    Generates a story using the API with retry logic.
    """
    if not model_id: return None

    retries = 3
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a creative writer. Write a compelling short story (300-500 words) based on the prompt."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85,
                max_tokens=1000,
                timeout=90
            )
            content = response.choices[0].message.content
            if content: return content

        except Exception as e:

            error_str = str(e)
            wait_time = 5 * (i + 1)

            if "429" in error_str:

                time.sleep(wait_time)
            else:
                time.sleep(2)

    return None

def process_single_row(index, row):
    """
    Processing logic for a single row (Prompt).
    """
    prompt_text = row.get('prompt', row.iloc[0] if len(row) > 0 else "")
    human_text = row.get('story', row.iloc[1] if len(row) > 1 else "")

    row_data = {
        "id": index,
        "prompt": prompt_text,
        "human_story": human_text
    }

    for name in ["gpt_mini", "gemini_flash", "claude_haiku"]:
        row_data[f"ai_{name}"] = generate_story(prompt_text, MODELS[name])

    if index < SOTA_SAMPLE_SIZE:
        for name in ["gemini_pro_sota", "claude_sonnet_sota"]:
            row_data[f"ai_{name}"] = generate_story(prompt_text, MODELS[name])
    else:
        row_data["ai_gemini_pro_sota"] = None
        row_data["ai_claude_sonnet_sota"] = None

    return row_data

def main():
    print(f"--- Starting Data Collection Script ---")
    print(f"Config: Threads={MAX_WORKERS}, Strategy=Base({TOTAL_SAMPLE_SIZE}) / SOTA({SOTA_SAMPLE_SIZE})")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded input file: {INPUT_CSV_PATH} ({len(df)} rows).")
    except FileNotFoundError:
        print(f" ERROR: Could not find file {INPUT_CSV_PATH}. Please check the filename.")
        sys.exit()

    if len(df) > TOTAL_SAMPLE_SIZE:
        print(f"Randomly sampling {TOTAL_SAMPLE_SIZE} prompts...")
        df = df.sample(n=TOTAL_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    headers = [
        "id", "prompt", "human_story",
        "ai_gpt_mini", "ai_gemini_flash", "ai_claude_haiku",
        "ai_gemini_pro_sota", "ai_claude_sonnet_sota"
    ]
    pd.DataFrame(columns=headers).to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Created output file: {OUTPUT_CSV_PATH}")

    print("Starting generation... (Please watch the progress bar)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(process_single_row, index, row): index
            for index, row in df.iterrows()
        }
        for future in tqdm(as_completed(future_to_index), total=len(df), desc="Processing"):
            try:
                data = future.result()

                with csv_lock:
                    pd.DataFrame([data]).to_csv(OUTPUT_CSV_PATH, mode='a', header=False, index=False)

            except Exception as exc:
                print(f"\nTask failed with error: {exc}")

    print(f"\n All done! Data saved to: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()