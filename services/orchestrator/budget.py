import os
from typing import NamedTuple

class Budget(NamedTuple):
    draft_max: int
    reflect_max: int
    revise_max: int
    learned_max: int
    total_context: int

def get_budget() -> Budget:
    """
    Reads environment variables to establish token budgets for each PB2S step.
    Defaults to heuristic ratios if not set.
    """
    total = int(os.getenv("PB2A_CONTEXT_TOKENS", "4096"))
    
    # Ratios
    r_draft = float(os.getenv("PB2A_DRAFT_RATIO", "0.20"))
    r_reflect = float(os.getenv("PB2A_REFLECT_RATIO", "0.25"))
    r_revise = float(os.getenv("PB2A_REVISE_RATIO", "0.50"))
    r_learned = float(os.getenv("PB2A_LEARNED_RATIO", "0.05"))
    
    # Normalize if sum > 1.0 (or significantly off)
    total_ratio = r_draft + r_reflect + r_revise + r_learned
    if total_ratio <= 0:
        # Fallback defaults
        r_draft, r_reflect, r_revise, r_learned = 0.2, 0.25, 0.5, 0.05
        total_ratio = 1.0
        
    scale = 1.0 / total_ratio
    
    return Budget(
        draft_max=int(total * r_draft * scale),
        reflect_max=int(total * r_reflect * scale),
        revise_max=int(total * r_revise * scale),
        learned_max=int(total * r_learned * scale),
        total_context=total
    )

def estimate_tokens(text: str) -> int:
    """
    Heuristic estimator (no nuktik/tiktoken).
    Roughly 1 token ~= 4 chars typically, but let's be conservative.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
