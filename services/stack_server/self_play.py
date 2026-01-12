import os
import time
import requests

# Configuration
STACK_SERVER_URL = os.getenv("STACK_SERVER_URL", "http://localhost:8000")
# In a real scenario, this would point to the LLM service (e.g., vLLM or text-generation-inference)
# For this 'Self-Play' sim, we will mock the LLM or use a placeholder if the orchestrator isn't running.
LLM_SERVICE_URL = "http://localhost:8000/generate" 

SEED_TOPICS = [
    "Calculate 25 + 25 = ?",
    "Explain why the sky is blue",
    "Calculate 10 * 10 = ?"
]

def mock_llm_generate(prompt, reflection=None):
    """
    Simulates a 'Dumb' model that makes mistakes, to test the CAE.
    """
    if "25 + 25" in prompt:
        return "The answer is 25 + 25 = 60." # Intentional Error
    if "10 * 10" in prompt:
        return "The answer is 10 * 10 = 100." # Correct
    if reflection:
        return f"Revised answer: {prompt[:20]} with corrections applied."
    return f"This is a draft explanation about: {prompt[:20]}..."

def run_self_play_loop():
    print("Starting PB2S Self-Play Simulation...")
    
    for topic in SEED_TOPICS:
        print(f"\n[TOPIC]: {topic}")
        
        # 1. Draft Phase (Model)
        draft = mock_llm_generate(topic)
        print(f"[DRAFT]: {draft}")

        # 2. Audit Phase (Stack Server)
        try:
            response = requests.post(
                f"{STACK_SERVER_URL}/audit",
                json={"draft_text": draft, "metadata": {"prompt": topic}}
            )
            audit_data = response.json()
        except Exception as e:
            print(f"Error contacting Stack Server: {e}")
            continue
            
        print(f"[CAE Score]: {audit_data['score']}")
        
        # 3. Reflection Phase (IRQ)
        reflection = audit_data["reflection_prompt"]
        learned = audit_data.get("learned_rule", "")
        print(f"[REFLECT]: {reflection}")
        print(f"[LEARNED]: {learned}")
        
        # 4. Revise Phase (Model)
        revise = mock_llm_generate(topic, reflection=reflection)
        print(f"[REVISE]: {revise}")
        
        # 5. Store Memory + Trace
        trace_text = (
            f"DRAFT:\n{draft}\n\n"
            f"REFLECT:\n{reflection}\n\n"
            f"REVISE:\n{revise}\n\n"
            f"LEARNED:\n{learned}"
        )
        try:
            requests.post(
                f"{STACK_SERVER_URL}/memory/add",
                json={"learned": learned, "metadata": {"prompt": topic}}
            )
            requests.post(
                f"{STACK_SERVER_URL}/trace/store",
                json={
                    "prompt": topic,
                    "draft": draft,
                    "reflect": reflection,
                    "revise": revise,
                    "learned": learned,
                    "scores": {"draft_score": audit_data.get("score", 0.0)},
                    "metadata": {"trace": trace_text}
                }
            )
            query_resp = requests.post(
                f"{STACK_SERVER_URL}/memory/query",
                json={"prompt": topic, "top_k": 2}
            )
            print(f"[MEMORY]: {query_resp.json().get('results')}")
        except Exception as e:
            print(f"Error storing trace/memory: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    # Ensure server is running or this script is run where it can reach it
    run_self_play_loop()
