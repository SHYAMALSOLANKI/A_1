from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os
import json
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("TokenizerTrainer")

TRAINING_FILE = "training_data/curriculum.jsonl"
OUTPUT_DIR = "services/tokenizer"
VOCAB_SIZE = 8192

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_tokenizer():
    logger.info(f"Training Custom BPE Tokenizer (Vocab: {VOCAB_SIZE}) on {TRAINING_FILE}...")
    
    # Initialize Tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Customize Special Tokens
    # Crucial: Add 'DRAFT:', 'REFLECT:', 'REVISE:' as atomic tokens so the model sees them as single concepts
    special_tokens = [
        "<s>", "</s>", "<pad>", "<unk>", 
        "DRAFT:", "REFLECT:", "REVISE:", "LEARNED:", 
        "[MATH]", "[LOGIC]", "[GRAMMAR]"
    ]
    
    # Train
    # We yield lines from the jsonl file
    def iterate_data():
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        yield obj.get("text", "")
                    except:
                        pass

    tokenizer.train_from_iterator(
        iterate_data(), 
        vocab_size=VOCAB_SIZE, 
        min_frequency=2, 
        show_progress=True,
        special_tokens=special_tokens
    )
    
    # Save
    logger.info(f"Saving tokenizer to {OUTPUT_DIR}")
    tokenizer.save_model(OUTPUT_DIR)
    
    # Create simple config to make it loadable by Transformers
    config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": 1024
    }
    with open(os.path.join(OUTPUT_DIR, "tokenizer_config.json"), "w") as f:
        json.dump(config, f)
        
    logger.info("Tokenizer training complete.")

if __name__ == "__main__":
    train_tokenizer()
