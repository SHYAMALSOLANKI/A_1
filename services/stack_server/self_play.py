import os
import requests
import json
import time

# Configuration
STACK_SERVER_URL = "http://localhost:9000"
# In a real scenario, this would point to the LLM service (e.g., vLLM or text-generation-inference)
# For this 'Self-Play' sim, we will mock the LLM or use a placeholder if the orchestrator isn't running.
LLM_SERVICE_URL = "http://localhost:8000/generate" 

SEED_TOPICS = [
    "Calculate 25 + 25 = ?",
    "Explain why the sky is blue",
    "Calculate 10 * 10 = ?"
]

def mock_llm_generate(prompt):
    """
    Simulates a 'Dumb' model that makes mistakes, to test the CAE.
    """
    if "25 + 25" in prompt:
        return "The answer is 25 + 25 = 60." # Intentional Error
    if "10 * 10" in prompt:
        return "The answer is 10 * 10 = 100." # Correct
    return f"This is a draft explanation about: {prompt[:20]}..."

def run_self_play_loop():
    print("Starting PB2S Self-Play Simulation...")
    
    for topic in SEED_TOPICS:
        print(f"\n[TOPIC]: {topic}")
        
        # 1. Draft Phase (Model)
        # draft = requests.post(LLM_SERVICE_URL, json={"prompt": topic}).json()['text']
        draft = mock_llm_generate(topic)
        print(f"[DRAFT]: {draft}")
        
        # 2. Audit Phase (Stack Server)
        try:
            response = requests.post(f"{STACK_SERVER_URL}/audit", json={"draft_text": draft})
            audit_data = response.json()
        except Exception as e:
            print(f"Error contacting Stack Server: {e}")
            continue
            
        print(f"[CAE Score]: {audit_data['score']}")
        
        # 3. Reflection Phase (IRQ)
        reflection = audit_data['reflection_prompt']
        print(f"[IRQ Injection]: {reflection}")
        
        # 4. Revise Phase (Model)
        # The trainer would now call the model again with (Draft + Reflection)
        # revise = requests.post(LLM_SERVICE_URL, json={"prompt": topic + reflection}).json()
        
        if not audit_data['passed']:
            print("[STATUS]: FAIL -> REVISION REQUIRED")
        else:
            print("[STATUS]: PASS -> STORE TRACE")
            
        time.sleep(1)

if __name__ == "__main__":
    # Ensure server is running or this script is run where it can reach it
    run_self_play_loop()
