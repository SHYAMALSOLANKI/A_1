
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
import os

def check_inference():
    output_dir = "./drrl-scratch-output/checkpoint-22200"
    
    # Check if checkpoint exists
    if not os.path.exists(output_dir):
        print(f"Error: {output_dir} not found.")
        return

    print(f"Loading model from {output_dir}...")
    
    # Load Tokenizer (SmolLM2-360M-Instruct used in training)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    
    # Load Model
    model = LlamaForCausalLM.from_pretrained(output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Test Prompts
    prompts = [
        "User: Hi.\n",
        "User: Explain gravity.\n"
    ]
    
    print("\n" + "="*50)
    print("INFERENCE CHECK (Step ~22250 / 46%)")
    print("="*50)

    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        print(f"\nPROMPT: {p.strip()}")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                do_sample=True, 
                temperature=0.6,    # Lower temp for stability
                repetition_penalty=1.2
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"OUTPUT:\n{generated}")
        print("-" * 30)

if __name__ == "__main__":
    check_inference()
