
import os
import yaml
import asyncio
import torch
from typing import List, Dict
from sqlalchemy import select
from sqlalchemy.orm import selectinload

# Extensions for training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import Dataset

# Internal packages
from memory.db import AsyncSessionLocal
from memory.models import Turn, Draft, Revision

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

async def fetch_training_data() -> List[Dict[str, str]]:
    """
    Queries the PB2A database for high-quality traces.
    We select Turns that have a Revision (meaning they went through the full loop).
    Data Format: 'text' field for SFT containing "User -> Revision" mapping.
    """
    print("Connecting to Postgres to fetch training dataset...")
    async with AsyncSessionLocal() as session:
        # Join Turn -> Draft -> Revision
        stmt = (
            select(Turn)
            .join(Draft)
            .join(Revision)
            .options(selectinload(Turn.draft).selectinload(Draft.revision))
        )
        result = await session.execute(stmt)
        turns = result.scalars().all()
        
        dataset = []
        for t in turns:
            if t.draft and t.draft.revision:
                # Format: Simple instruction tuning (distillation of the revision)
                # You could also include the "DRAFT..REFLECT" steps if you want Process Cloning.
                row = {
                    "text": f"### User:\n{t.user_input}\n\n### Assistant:\n{t.draft.revision.text}"
                }
                dataset.append(row)
        
        print(f"Fetched {len(dataset)} valid training examples from DB.")
        return dataset

def train(config_path: str):
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)

    # 1. Load Data
    raw_data = asyncio.run(fetch_training_data())
    if not raw_data:
        print("No training data found. Run more conversations in Orchestrator first!")
        return
    
    hf_dataset = Dataset.from_list(raw_data)

    # 2. Setup Model & Tokenizer
    model_name = cfg.get("model_name", "meta-llama/Llama-2-7b-hf")
    print(f"Loading model: {model_name}")
    
    # NOTE: In a real container with GPU, remove device_map="cpu" or use "auto"
    # We use "cpu" here just to ensure it runs in this constrained check environment if needed,
    # but for real training, ensure you have nvidia-runtime enabled.
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        trust_remote_code=True
    )

    # 3. PEFT Config
    peft_conf = cfg.get("peft", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_conf.get("r", 8),
        lora_alpha=peft_conf.get("lora_alpha", 16),
        lora_dropout=peft_conf.get("lora_dropout", 0.05),
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=cfg.get("output_dir", "./adapter-output"),
        per_device_train_batch_size=cfg.get("batch_size", 1),
        num_train_epochs=cfg.get("epochs", 1),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        logging_steps=10,
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available()
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 512),
        tokenizer=tokenizer,
        args=args,
    )

    print("Starting Training...")
    trainer.train()
    
    # 6. Save
    output_dir = cfg.get("output_dir", "./adapter-output")
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)
    print("Training job complete.")

if __name__ == "__main__":
    train("config.yaml")
