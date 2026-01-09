from pb2s_prompts import get_system_prompt

def run_eval(task_name: str, verbose: bool = False):
    """
    Runs an evaluation task using prompts.
    """
    # Example logic using the prompt package
    prompt = get_system_prompt()
    if verbose:
        print(f"Running eval for: {task_name}")
        print(f"Using System Prompt: {prompt[:50]}...")
    
    # Placeholder for actual eval logic (e.g. calling an LLM)
    return {"status": "success", "task": task_name, "score": 0.95} 
