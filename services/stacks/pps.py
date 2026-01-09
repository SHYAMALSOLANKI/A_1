import re

class PPS:
    """
    Persistent Prompt Stack (PPS)
    Acts as the 'Constitution' or static memory for the agent.
    """
    
    def __init__(self):
        self.core_identity = "You are a reasoner, not a simple text predictor. You must allow for internal debate."
        self.safety_policy = "Do not generate hate speech, violent content, or illegal acts."
        
        # Invariant Rules (Axioms)
        self.axioms = {
            "syntax": "Output must be plain text or Markdown, not raw LaTeX source code.",
            "logic": "Conclusions must follow from the premises provided.",
            "completeness": "Drafts must not cut off mid-sentence."
        }

    def get_system_prompt(self) -> str:
        """Returns the foundational system prompt."""
        return f"{self.core_identity}\nSafety: {self.safety_policy}"

    def get_axiom(self, key: str) -> str:
        return self.axioms.get(key, "")
