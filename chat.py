import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

def get_latest_checkpoint(base_dir):
    checkpoints = glob.glob(os.path.join(base_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Sort by step number (integer after 'checkpoint-')
    checkpoints.sort(key=lambda x: int(x.split("checkpoint-")[-1]), reverse=True)
    return checkpoints[0]

def chat():
    base_output_dir = "./drrl-scratch-output"
    
    print("Searching for latest checkpoint...")
    checkpoint_dir = get_latest_checkpoint(base_output_dir)
    
    if not checkpoint_dir:
        print(f"No checkpoints found in {base_output_dir}")
        return

    print(f"Loading model from: {checkpoint_dir}")
    
    # Load Tokenizer & Model
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        model = LlamaForCausalLM.from_pretrained(checkpoint_dir)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Moving model to {device}...")
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\n" + "="*50)
    print("🤖 MODEL CHAT INTERFACE")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    history = ""

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                break
            
            if not user_input:
                continue

            # Force the training trigger phrase to activate the PB2S behaviors
            prompt = f"User: {user_input}\nRun PB2S loop.\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,  # Increased to allow full loop
                    do_sample=True,
                    temperature=0.6,     # Slightly lower temp to reduce "dots" loop
                    repetition_penalty=1.3, # Higher penalty to kill the "......"
                    pad_token_id=tokenizer.eos_token_id
                )

            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new part
            response = full_output[len(prompt):].strip()
            
            print(f"Model: {response}\n")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat()
