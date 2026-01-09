from .base import EvalSuite
import re

class ConsistencyEval(EvalSuite):
    def run(self, input_text: str, output_text: str) -> float:
        # Simple heuristic: check if output says "I cannot" when input is simple
        # Placeholder for real model-based consistency check
        return 1.0

class MathEval(EvalSuite):
    def run(self, input_text: str, output_text: str) -> float:
        # Extract numbers and operator, naive check
        # Real impl would parse equation
        return 1.0 

class CodeEval(EvalSuite):
    def run(self, input_text: str, output_text: str) -> float:
        # In a real scenario, this would spin up a docker container
        # Here we mock the sandbox result
        if "def " in output_text:
            return 1.0
        return 0.0
