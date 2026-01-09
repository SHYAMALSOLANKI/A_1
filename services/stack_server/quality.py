from collections import deque
import statistics
from typing import Dict, Any

class DynamicQualityController:
    """
    DQC: Global Adaptive Feedback Regulator.
    Adjusts Beta (Penalty Weight) and Thresholds based on Model Performance.
    """
    def __init__(self, window_size=200):
        self.window_size = window_size
        self.scores = {
            "math": deque(maxlen=window_size),
            "grammar": deque(maxlen=window_size),
            "logic": deque(maxlen=window_size),
            "global": deque(maxlen=window_size)
        }
        
        # Initial Parameters
        self.beta = 0.2  # Start with moderate penalty
        self.thresholds = {
            "math": 0.9,    # Math must be strict
            "grammar": 0.8, # Grammar can handle minor errors
            "logic": 0.75   # Logic is hardest, start lenient
        }
        
        # Difficulty Level (0.0 to 1.0)
        self.curriculum_difficulty = 0.1 

    def update(self, rule_type: str, score: float):
        """
        Ingest a new score (0.0 - 1.0) from the CAE.
        """
        # Categorize rule type if generic
        category = "global"
        if "math" in rule_type.lower(): category = "math"
        elif "grammar" in rule_type.lower(): category = "grammar"
        elif "logic" in rule_type.lower(): category = "logic"
        
        self.scores[category].append(score)
        self.scores["global"].append(score)
        
        self._adapt_parameters()

    def _adapt_parameters(self):
        """
        The Brain: Adjusts Beta and Thresholds based on moving averages.
        """
        global_avg = self.global_mean()
        
        # 1. Adapt Beta (Penalty Weight)
        # If model is failing (low score), reduce penalty to prevent gradient explosion.
        # If model is succeeding, increase penalty to force perfection.
        # Clamp between 0.05 and 0.5
        target_beta = 0.2 + (global_avg * 0.3) 
        self.beta = max(0.05, min(0.5, target_beta))

        # 2. Adapt Curriculum
        # If we are aceing it (>0.9), make it harder.
        if global_avg > 0.9:
            self.curriculum_difficulty = min(1.0, self.curriculum_difficulty + 0.001)
        elif global_avg < 0.7:
            self.curriculum_difficulty = max(0.1, self.curriculum_difficulty - 0.005)

    def get_beta(self) -> float:
        return self.beta

    def get_threshold(self, rule_type: str) -> float:
        # Simple mapping for now
        if "math" in rule_type.lower(): return self.thresholds["math"]
        if "grammar" in rule_type.lower(): return self.thresholds["grammar"]
        return 0.8 # Default

    def global_mean(self) -> float:
        if not self.scores["global"]: return 0.5
        return statistics.mean(self.scores["global"])

    def report_metrics(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "difficulty": self.curriculum_difficulty,
            "avg_global": self.global_mean(),
            "avg_math": statistics.mean(self.scores["math"]) if self.scores["math"] else 0.0
        }
