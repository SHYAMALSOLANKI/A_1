
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

def check_model_intelligence():
    base_output_dir = "./drrl-scratch-output/checkpoint-22200"
    
    print(f"DIAGNOSTIC: Loading model from {base_output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    model = LlamaForCausalLM.from_pretrained(base_output_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # We will test two very different inputs to see if the "Dumbness" is universal or prompt-dependent.
    
    # TEST 1: The "Triggered" Prompt (What it was trained on)
    prompt_trained = "User: Define the concept of Gravity.\nRun PB2S loop.\n"
    
    # TEST 2: The "Natural" Prompt (What allows it to hallucinate freely)
    prompt_natural = "The history of the Roman Empire is" 

    prompts = [prompt_trained, prompt_natural]

    print("\n" + "="*50)
    print("INTELLIGENCE CHECK")
    print("="*50)

    for p in prompts:
        print(f"\nINPUT: {p.strip()}")
        inputs = tokenizer(p, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.6,
                repetition_penalty=1.2
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"OUTPUT:\n{generated[len(p):].strip()}")
        print("-" * 30)

if __name__ == "__main__":
    check_model_intelligence()
