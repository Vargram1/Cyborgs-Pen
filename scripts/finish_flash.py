import pandas as pd
from openai import OpenAI
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import sys
from keys import OPENROUTER_API_KEY

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE_PATH = "data/final_dataset_human_vs_ai.csv"

GEMINI_FLASH_ID = "google/gemini-2.0-flash-001"

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
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a creative writer. Write a compelling short story (300-500 words) based on the prompt."},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.85, 
                max_tokens=1000, 
                timeout=60
            )
            return response.choices[0].message.content
        except Exception:
            time.sleep(2)
    return None

def repair_row(index, row):
    """
    Checks if the Flash column is empty and generates content if needed.
    """
    prompt_text = row['prompt']
    updates = {}
    
    if pd.isna(row['ai_gemini_flash']):
        updates['ai_gemini_flash'] = generate_story(prompt_text, GEMINI_FLASH_ID)
        
    return index, updates


def main():
    print("--- Starting Script: Finish Gemini Flash Only ---")
    
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded CSV: {len(df)} rows.")
    except FileNotFoundError:
        print("ERROR: Could not find the CSV file.")
        return

    tasks = []
    for idx, row in df.iterrows():
        if pd.isna(row['ai_gemini_flash']):
            tasks.append((idx, row))
            
    print(f"Found {len(tasks)} rows missing Gemini Flash data.")
    print("Starting generation... (Gemini 3 Pro is disabled and will be ignored)")
    
    with ThreadPoolExecutor(max_workers=10) as executor: 
        future_to_idx = {executor.submit(repair_row, idx, row): idx for idx, row in tasks}
        
        for future in tqdm(as_completed(future_to_idx), total=len(tasks), desc="Filling Flash"):
            idx, updates = future.result()
            
            if updates:
                with csv_lock:
                    for col, val in updates.items():
                        df.at[idx, col] = val
                    
                    df.to_csv(CSV_FILE_PATH, index=False)

    print("\nGemini Flash completion finished! The dataset is now ready for cleaning.")

if __name__ == "__main__":
    main()