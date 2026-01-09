import torch
import sys
import os

sys.path.append(os.getcwd())

from services.model.reasoning_network import RecurrentReasoningNetwork, ReasoningConfig

# Configuration from train_drrl.py
config = ReasoningConfig(
    vocab_size=50257, # Assuming GPT2 size as fallback/baseline since tokenizer is dynamic
    hidden_size=512,        
    num_hidden_layers=24,    
    num_reasoning_loops=3,  
    num_attention_heads=8,
    intermediate_size=1376,
    max_position_embeddings=1024
)

model = RecurrentReasoningNetwork(config)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Architecture Depth: {config.num_hidden_layers} physical layers x {config.num_reasoning_loops} loops")
