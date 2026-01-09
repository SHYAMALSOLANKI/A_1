import os
import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Suppress warnings
warnings.filterwarnings("ignore")

def chat_with_checkpoint(checkpoint_path: str):
    print(f"Loading model from: {checkpoint_path}")
    print("This may take a moment...")

    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on: {device.upper()}")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        print("\n\n" + "="*50)
        print("CHECKPOINT LOADED. READY TO CHAT.")
        print("Type 'exit' or 'quit' to stop.")
        print("="*50 + "\n")

        system_prompt = "You are an intelligent assistant. Answer the user's questions step-by-step."

        while True:
            user_input = input("\nUSER: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # Simple chat format without the complex PB2S framing, just to check emergence
            prompt = f"System: {system_prompt}\nUser: {user_input}\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print("ASSISTANT: ", end="", flush=True)
            _ = model.generate(
                **inputs, 
                streamer=streamer, 
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            print("")

    except Exception as e:
        print(f"\nERROR: Failed to load or run model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the absolute path provided by the user
    CKPT_PATH = r"C:\Users\Shyamal solanki\A_1\checkpoint-epoch-3"
    if os.path.exists(CKPT_PATH):
        chat_with_checkpoint(CKPT_PATH)
    else:
        print(f"Checkpoint not found at: {CKPT_PATH}")
