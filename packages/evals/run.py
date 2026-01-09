import argparse
from evals.suites import ConsistencyEval, MathEval, CodeEval

def main():
    parser = argparse.ArgumentParser(description="Run PB2A Evals")
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--suite", type=str, choices=["all", "math", "code", "consistency"], default="all")
    
    args = parser.parse_args()
    
    print(f"Running evals for model {args.model_version} on suite {args.suite}")
    # Mock run
    print("Score: 0.98")

if __name__ == "__main__":
    main()
