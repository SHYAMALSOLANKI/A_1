import time
from evals import run_eval
from memory import init_db

async def run_training_job():
    print("Starting training job...")
    await init_db()
    
    # Simulate training
    time.sleep(2)
    print("Training complete.")

    # Run eval
    print("Running post-training evaluation...")
    result = run_eval("summarization_task", verbose=True)
    print(f"Eval result: {result}")

if __name__ == "__main__":
    # In production, we would use an async runner or job queue (e.g. celery)
    # For now, we just invoke the training script directly to test data connectivity.
    from train import train
    train("config.yaml")
