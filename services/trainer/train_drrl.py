import os
import yaml
import torch
import logging
import json
from torch.utils.data import Dataset, DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    AdamW, 
    AutoTokenizer
)
from tqdm import tqdm
import sys

# Add workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from services.model.reasoning_network import RecurrentReasoningNetwork, ReasoningConfig
from services.model.pb2s import PB2SModel

# Configure Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Configuration ---
STACK_SERVER_URL = os.getenv("STACK_SERVER_URL", "http://localhost:8000")
TOKENIZER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer"))

# Default Hyperparameters
DEFAULT_CONFIG = {
    "learning_rate": 3e-4,
    "num_train_epochs": 3,
    "dataset_paths": ["training_data/curriculum.jsonl"],
    "output_dir": "./drrl_output"
}

class DRRLDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        if not os.path.exists(file_path):
            logger.warning(f"Dataset not found at {file_path}")
            return
            
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", obj.get("prompt", ""))
                        if text:
                            self.data.append(text)
                    except json.JSONDecodeError:
                        pass
        logger.info(f"Loaded {len(self.data)} samples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_config(config_path=None):
    if config_path is None:
        # Default to drrl_config.yaml in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), "drrl_config.yaml")
        
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"Config file {config_path} not found. Using defaults.")
    return DEFAULT_CONFIG

def train_drrl():
    logger.info("Initializing PB2S-DRRL Trainer...")
    
    # 1. Load Configuration
    config = load_config()
    
    # 2. Initialize Tokenizer
    logger.info(f"Loading Custom Tokenizer from {TOKENIZER_PATH}")
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH)
        tokenizer.pad_token = "<pad>"
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    except Exception as e:
        logger.error(f"Failed to load custom tokenizer: {e}. Falling back to GPT2.")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Initialize Neural Network (System 1)
    logger.info("Initializing Recurrent Reasoning Network (The Brain)")
    # Load hyperparams from Config or Default
    m_conf = config.get("model_config", {})
    reasoning_config = ReasoningConfig(
        vocab_size=m_conf.get("vocab_size", len(tokenizer)),
        hidden_size=m_conf.get("hidden_size", 512),        
        num_hidden_layers=m_conf.get("num_hidden_layers", 24),    
        num_reasoning_loops=3,  
        num_attention_heads=m_conf.get("num_attention_heads", 8),
        intermediate_size=m_conf.get("intermediate_size", 2048),
        max_position_embeddings=m_conf.get("max_position_embeddings", 2048)
    )
    raw_model = RecurrentReasoningNetwork(reasoning_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_model.to(device)
    logger.info(f"Model initialized on {device}. Effective Depth: {reasoning_config.num_hidden_layers * 3} layers.")
    
    # 4. Initialize PB2S Controller (The Loop Manager)
    # We pass the optimizer later when we create it
    
    # 5. Data & Optimizer
    # Load all datasets indicated in config sequentially or merged
    dataset_paths = config.get("dataset_paths", [])
    if isinstance(dataset_paths, str): dataset_paths = [dataset_paths]
    
    full_dataset_data = []
    for d_path in dataset_paths:
        logger.info(f"Loading dataset: {d_path}")
        ds = DRRLDataset(d_path)
        full_dataset_data.extend(ds.data)
        
    if len(full_dataset_data) == 0:
        logger.error("No data found in any provided paths. Exiting.")
        return

    # Create a unified dataset from all loaded data
    # (Note: For true curriculum learning, we should loop through datasets in the outer loop,
    # but for simplicity/robustness we merge them sequence-preserving if DataLoader is not shuffled effectively.
    # However, standard practice often shuffles. If user wants strict 1->2->3 order, we need a custom sampler or loop.)
    
    # Implementing strict Curriculum Order loop
    # We will ignore the merged list above and loop properly below
    
    optimizer = AdamW(raw_model.parameters(), lr=float(config.get("learning_rate", 3e-4)))
    pb2s = PB2SModel(raw_model, tokenizer, backend_url=STACK_SERVER_URL, optimizer=optimizer, device=device)
    
    global_step = 0
    total_epochs = config.get("num_train_epochs", 1)
    
    for epoch in range(total_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{total_epochs}")
        
        # Iterate through curriculum phases
        for d_idx, d_path in enumerate(dataset_paths):
            logger.info(f"--- Curriculum Phase {d_idx+1}: {d_path} ---")
            phase_ds = DRRLDataset(d_path)
            if len(phase_ds) == 0: continue
            
            phase_loader = DataLoader(phase_ds, batch_size=config.get("batch_size", 1), shuffle=True)
            progress_bar = tqdm(phase_loader, desc=f"Phase {d_idx+1} ({os.path.basename(d_path)})")
            
            for batch_prompts in progress_bar:
                prompt = batch_prompts[0]
                if "\nDRAFT:" in prompt:
                    prompt = prompt.split("\nDRAFT:")[0].strip()

                # --- FULL CYCLE: DRAFT -> REFLECT -> REVISE -> LEARN ---
                trace = pb2s.run_cycle(prompt)

                loss = trace.get("loss", 0.0)
                score = trace.get("score", 0.0)

                global_step += 1
                if global_step % 5 == 0:
                    progress_bar.set_postfix({
                        "Loss": f"{loss:.2f}",
                        "Score": f"{score:.2f}",
                    })
                
    # Save
    out_dir = config.get("output_dir", "./drrl_output")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(raw_model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(out_dir)
    logger.info(f"Training complete. Model saved to {out_dir}")

if __name__ == "__main__":
    train_drrl()
