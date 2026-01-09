from abc import ABC, abstractmethod

class EvalSuite(ABC):
    @abstractmethod
    def run(self, input_text: str, output_text: str) -> float:
        """Returns a score between 0.0 and 1.0"""
        pass
