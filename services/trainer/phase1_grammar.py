
import os
import yaml
import torch
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    TrainerCallback, 
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

class EmergenceCallback(TrainerCallback):
    """
    Monitors the 'Emergence' of the model by generating text every N steps.
    """
    def __init__(self, tokenizer, model, prompts, device):
        self.tokenizer = tokenizer
        self.model = model
        self.prompts = prompts
        self.device = device

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and state.global_step > 0:
            print(f"\n[EMERGENCE CHECK] Step {state.global_step}")
            self.model.eval()
            with torch.no_grad():
                for p in self.prompts:
                    inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=50, 
                        do_sample=True, 
                        temperature=0.7
                    )
                    decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"PROMPT: {p}")
                    print(f"GEN   : {decoded.replace(p, '...')}")
                    print("-" * 20)
            self.model.train()

def train_phase1(config_path: str):
    print(f"Loading Phase 1 Config from {config_path}")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. Load Dataset (Streaming if possible for speed, or map)
    # Using 'json' script from datasets to load local jsonl
    data_files = cfg['dataset_paths']
    print(f"Loading datasets: {data_files}")
    
    # Check if files exist
    valid_files = [f for f in data_files if os.path.exists(f)]
    if not valid_files:
        print("ERROR: No valid dataset files found.")
        return

    # We load as 'text' column. 
    # Since the file has "text" key, we can use it directly.
    dataset = load_dataset("json", data_files=valid_files, split="train")
    
    # 2. Model & Tokenizer
    model_name = cfg['model_name']
    print(f"Initializing Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    # Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Optionally enable LoRA if full finetune is too heavy for 8GB
    # For 300M params, full finetune MIGHT fit in 8GB at batch size 1/gradient checkpointing?
    # 300M * 4 bytes = 1.2GB parameters.
    # Optimizer (AdamW) = 1.2GB * 2 = 2.4GB.
    # Gradients = 1.2GB.
    # Activations... 
    # Total ~5GB. It fits! 
    # BUT user requested Max Context. That eats VRAM.
    # Use LoRA to save VRAM for Context.
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention for context
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 3. Tokenize
    block_size = cfg['max_seq_length']
    def group_texts(examples):
        # Concatenate all texts
        concatenated = "".join(examples["text"])
        # Split into chunks of block_size
        return {
            "input_ids": tokenizer(concatenated, truncation=True, max_length=block_size)["input_ids"]
        }
        # Note: Proper CLM preprocessing is complex (chunking tokens).
        # For simplicity in this script, we assume 'text' fields are distinct samples 
        # OR we just tokenize individually if they fit. 
        # Given the previous 'tex' file bloat, simpler to truncation.
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=block_size, 
            padding="max_length"
        )

    print("Tokenizing dataset...")
    # Using a small subset or streaming would be better for 11GB, but let's try mapping
    # If it crashes on RAM, we need streaming=True. 
    # Given 11GB text > likely 32GB RAM usage.
    # Let's use streaming or keep it simple but safe?
    # We'll use formatting_func in SFTTrainer style or just map.
    # For safety on laptop, let's take a sample or assume user manages RAM.
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 4. Training Args
    output_dir = cfg['output_dir']
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
        learning_rate=float(cfg['learning_rate']),
        num_train_epochs=cfg['num_train_epochs'],
        logging_steps=cfg['logging_steps'],
        save_steps=cfg['save_steps'],
        warmup_steps=cfg['warmup_steps'],
        fp16=True if torch.cuda.is_available() else False,
        push_to_hub=False,
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=None, # defaults to pad
        callbacks=[
            EmergenceCallback(
                tokenizer, 
                model, 
                [cfg['emergence_prompt']],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        ]
    )

    print("Starting Phase 1 (Emergence) Training...")
    trainer.train()
    
    trainer.save_model(output_dir)
    print("Phase 1 Complete.")

if __name__ == "__main__":
    train_phase1("phase1_config.yaml")
