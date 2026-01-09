# Neuro-Symbolic PB2S Architecture Definition
# Version 1.0 (Reflective Scratch Learning)

"""
This architecture definition outlines the components required to build a 
Self-Correcting Reasoning Engine starting from random initialization (Tabula Rasa).

The core hypothesis is that "Reasoning" is an emergent property of a 
recurisve feedback loop between a Neural Generator and a Symbolic Auditor.
"""

# ==========================================
# 1. ARTIFACT DEFINITIONS
# ==========================================

ARTIFACTS = {
    "TOKENIZER": {
        "type": "ByteLevelBPE",
        "vocab_size": 8192,  # Compact vocabulary to force reasoning over memorization
        "special_tokens": [
            "<s>", "</s>", "<pad>", "<unk>", 
            "DRAFT:", "REFLECT:", "REVISE:", "LEARNED:",  # Control Tokens
            "[MATH]", "[LOGIC]", "[GRAMMAR]"              # Domain Tokens
        ],
        "rationale": "Small vocab ensures digits (0-9) are atomic tokens, enabling arithmetic learning."
    },

    "NEURAL_NETWORK": {
        "architecture": "LlamaForCausalLM (Custom Config)",
        "parameters": "58 Million",
        "config": {
            "hidden_size": 512,        # Narrow width (forcing abstraction)
            "intermediate_size": 1376, # standard 2.7x expansion
            "num_hidden_layers": 16,   # Deep network (depth > width for logic steps)
            "num_attention_heads": 8,  # 64-dim heads
            "max_position_embeddings": 1024,
            "vocab_size": 8192         # Matches tokenizer
        },
        "rationale": "High depth-to-width ratio mimics 'System 2' serial processing."
    },

    "BACKEND_SERVER": {
        "role": "The Teacher (Symbolic Auditor)",
        "components": [
            "CAE (Contradiction Audit Engine): SymPy + NLI + Grammar",
            "IRQ (Interactive Response Queue): Translates errors to prompts",
            "DQC (Dynamic Quality Controller): Adjusts difficulty (Beta)"
        ]
    }
}

# ==========================================
# 2. METHODOLOGY OF OPERATION (The Loop)
# ==========================================

"""
The system does NOT operate as a standard LLM (Input -> Output).
It operates as a Recurrent State Machine.
"""

OPERATION_CYCLE = [
    "STATE 1: DRAFTING",
    "   Input:  User Prompt",
    "   Action: Model generates 'Draft' token-by-token.",
    "   Output: Raw String (likely flawed).",
    
    "STATE 2: AUDITING (Symbolic Intervention)",
    "   Input:  Draft String",
    "   Action: Backend Server runs rigid checks (Math, Logic, Syntax).",
    "   Output: Violation Report (e.g., '2+2!=5').",
    
    "STATE 3: REFLECTION",
    "   Input:  Draft + Violation Report",
    "   Action: Model focuses on the Error Signal.",
    "   Output: 'Reflection' string (Plan for correction).",
    
    "STATE 4: REVISION",
    "   Input:  Reflection",
    "   Action: Model generates Final Answer.",
    "   Output: Revised String."
]

# ==========================================
# 3. METHODOLOGY OF TRAINING (Self-Correction)
# ==========================================

TRAINING_STRATEGY = {
    "Algorithm": "Recursive Self-Supervised Learning (RSSL)",
    "Objective": "Minimize distance between Draft and Validated Revision.",
    
    "Loss_Function": "Hybrid_Loss = CE(Revision, Truth) + Beta * Rule_Violation_Penalty",
    
    "Curriculum_Phases": [
        "Phase 1: Syntax & Grammar (Learning to speak)",
        "Phase 2: Simple Arithmetic (Learning consistency)",
        "Phase 3: Multi-step Logic (Learning implication)",
        "Phase 4: Abstract Reasoning (Emergence)"
    ],
    
    "Handling_Noise": "Early in training, revisions will also be bad. The Server's 'Beta' parameter starts low and ramps up as the model gains capability."
}
