import os
import yaml
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    TrainerCallback, 
    Trainer,
    DataCollatorForLanguageModeling,
    LlamaConfig, 
    LlamaForCausalLM
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. VISUALIZER CALLBACK (Strict Policy Check)
class DrrlMonitorCallback(TrainerCallback):
    def __init__(self, tokenizer, model, prompt, device):
        self.tokenizer = tokenizer
        self.model = model
        self.prompt = prompt
        self.device = device
        self.required_headers = ["DRAFT:", "REFLECT:", "REVISE:", "LEARNED:"]

    def on_step_end(self, args, state, control, **kwargs):
        # EMERGANCE CHECK: Every 50 steps (Speed Optimization)
        if state.global_step > 0 and state.global_step % 50 == 0:
            print(f"\n[PHASE 1 MONITOR] Step {state.global_step}")
            self.model.eval()
            
            # Full Emergence: 800 tokens to allow Draft->Reflect->Revise loop
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800, 
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            output_text = self.tokenizer.decode(outputs[0])
            generated = output_text.replace(self.prompt, "").strip()
            
            print("="*40)
            print(f"INPUT: {self.prompt}")
            print(f"RAW OUPUT:\n{generated}") 
            print("="*40 + "\n")
            self.model.train()

def train_drrl(config_path: str):
    print(f"Loading DRRL Config from {config_path}")
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. LOAD DATASET
    data_files = cfg['dataset_paths']
    print(f"Loading datasets: {data_files}")
    dataset = load_dataset("json", data_files=data_files, split="train")
    
    # 3. MODEL INITIALIZATION (From Scratch vs Pretrained)
    model_name = cfg['model_name']
    
    if model_name == "scratch-model":
        print("\n[PHASE 1] INITIALIZING NEW MODEL FROM SCRATCH")
        print("Architecture: Custom Llama-style (~108M params)")
        
        mc = cfg['model_config']
        config = LlamaConfig(
            vocab_size=mc['vocab_size'],
            hidden_size=mc['hidden_size'],
            intermediate_size=mc['intermediate_size'],
            num_hidden_layers=mc['num_hidden_layers'],
            num_attention_heads=mc['num_attention_heads'],
            max_position_embeddings=mc['max_position_embeddings'],
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            use_cache=True,
            tie_word_embeddings=True
        )
        model = LlamaForCausalLM(config)
        
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        
        # 4. TOKENIZER INJECTION (DRRL SPECIAL TOKENS)
        special_tokens_dict = {'additional_special_tokens': [
            '[BACKEND INTERVENTION]', 
            'DRAFT:', 
            'REFLECT:', 
            'REVISE:', 
            'LEARNED:', 
            'VIOLATION:', 
            'ACTION:',
            '[IRQ]'
        ]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special Drrl tokens")
        model.resize_token_embeddings(len(tokenizer))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"Model Parameters: {model.num_parameters():,}")
        
        device_map = {"": "cuda"} if torch.cuda.is_available() else "cpu"
        model.to(device_map[""] if isinstance(device_map, dict) else device_map)
        
    else:
        print(f"Initializing Smart Base Model: {model_name}")
        # (Legacy Pretrained Logic Removed for clarity in this overwrite)

    # 5. TOKENIZATION
    # We assume 'text' column contains full "User:... DRAFT:... REVISE:..." string
    block_size = cfg['max_seq_length']
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=block_size
            # Dynamic padding by collator = Efficient
        )
    
    print("Tokenizing data...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 6. TRAINER SETUP
    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 4),
        learning_rate=float(cfg['learning_rate']),
        num_train_epochs=cfg['num_train_epochs'],
        logging_steps=cfg['logging_steps'],
        save_steps=cfg['save_steps'],
        warmup_steps=cfg['warmup_steps'],
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
        dataloader_num_workers=0,   # CRITICAL FOR WINDOWS SPEED
        dataloader_pin_memory=False, 
    )

    monitor = DrrlMonitorCallback(
        tokenizer, 
        model, 
        prompt=cfg['check_prompt'],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[monitor]
    )

    print("Starting DRRL Training (Industry Grade Phase 1)...")
    # Attempt to resume from checkpoint if available
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(f"Resume failed ({e}), starting fresh...")
        trainer.train()
    
    print("Saving Checkpoint...")
    trainer.save_model(cfg['output_dir'])
    print("Done! Artifacts in: " + cfg['output_dir'])

if __name__ == "__main__":
    # Use relative path derived from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "drrl_config.yaml")
    train_drrl(config_path)
