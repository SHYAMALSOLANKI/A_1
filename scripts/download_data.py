import os
import json
import logging
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "training_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_to_jsonl(dataset_iter, output_filename, limit=None, text_column="text"):
    path = os.path.join(OUTPUT_DIR, output_filename)
    logger.info(f"Saving to {path}...")
    
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset_iter):
            text = item.get(text_column, "")
            if not text:
                # Try common alternatives
                text = item.get("content", item.get("prompt", item.get("question", "") + "\n" + item.get("answer", "")))
            
            if text:
                f.write(json.dumps({"text": text}) + "\n")
                count += 1
                if limit and count >= limit:
                    break
    logger.info(f"Saved {count} records to {output_filename}")

def download_english():
    logger.info("--- Downloading 1. English (FineWeb-Edu High Quality) ---")
    # Approx 500k samples ~ 0.5B tokens depending on length
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    # Filter for very high educational value
    filtered = (x for x in ds if x['score'] is not None and x['score'] >= 4.5)
    save_to_jsonl(filtered, "01_english.jsonl", limit=300000, text_column="text")

def download_maths():
    logger.info("--- Downloading 2. Maths (OpenMathInstruct + MetaMath) ---")
    # MetaMathQA
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    save_to_jsonl(ds, "02_maths.jsonl", limit=200000, text_column="query") 

def download_logic():
    logger.info("--- Downloading 3. Logic & Reasoning (SlimOrca) ---")
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)
    # Combine system, user, assistant for full context
    def format_orca(x):
        conversations = x.get("conversations", [])
        text = ""
        for c in conversations:
            text += f"{c['from']}: {c['value']}\n"
        return {"text": text}
    
    mapped_ds = (format_orca(x) for x in ds)
    save_to_jsonl(mapped_ds, "03_logic.jsonl", limit=150000)

def download_science():
    logger.info("--- Downloading 4-6. Physics, Biology, Chemistry (SciQ + Camel) ---")
    # Using SciQ for general science QA
    ds_sciq = load_dataset("sciq", split="train", streaming=True)
    
    def format_sciq(x):
        return {"text": f"Question: {x['question']}\nAnswer: {x['correct_answer']}\nSupport: {x['support']}"}
        
    mapped_sciq = (format_sciq(x) for x in ds_sciq)
    
    # We need more volume for 2-3B tokens, so we append some Science-heavy FineWeb if possible
    # For now, sticking to SciQ + some general instruct for science
    save_to_jsonl(mapped_sciq, "04_science.jsonl", limit=None) # All SciQ

def download_coding():
    logger.info("--- Downloading 8. Coding (The Stack Smol XS) ---")
    ds = load_dataset("bigcode/the-stack-smol-xs", data_dir="data/python", split="train", streaming=True)
    save_to_jsonl(ds, "05_coding.jsonl", limit=100000, text_column="content")

if __name__ == "__main__":
    logger.info("Starting Data Download for Curriculum Learning...")
    try:
        download_english()
        download_maths()
        download_logic()
        download_science()
        download_coding()
        logger.info("All datasets downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download datasets: {e}")
        import traceback
        traceback.print_exc()
