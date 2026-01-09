from typing import Dict

class IRQ:
    """
    Interactive Response Queue (IRQ)
    The 'Teacher'. Translates CAE findings into System Interrupt/Injection prompts.
    """
    
    def __init__(self, pps):
        self.pps = pps

    def generate_intervention(self, ca_result: Dict) -> str:
        """
        Takes the Audit Result and generates the injection strings.
        """
        if ca_result["status"] == "PASS":
            return "[BACKEND]: Audit Passed. Proceed to Finalize."
        
        violations = ca_result.get("violations", [])
        instructions = []
        
        for v in violations:
            if v == "LATEX_LEAKAGE":
                instructions.append(f"VIOLATION: {self.pps.get_axiom('syntax')}")
                instructions.append("ACTION: Strip all LaTeX commands. Convert to Markdown.")
            elif v == "DRAFT_TOO_SHORT":
                instructions.append(f"VIOLATION: {self.pps.get_axiom('completeness')}")
                instructions.append("ACTION: Expand the reasoning. Define terms.")
            elif v == "REPETITIVE_LOOP":
                 instructions.append("VIOLATION: Low information density.")
                 instructions.append("ACTION: Do not repeat sentences. Provide new information.")
            elif v == "LOGIC_ERROR_MATH":
                 instructions.append(f"VIOLATION: {self.pps.get_axiom('logic')}")
                 instructions.append("ACTION: Recalculate step-by-step.")
                 
        intervention_prompt = (
            "\n[BACKEND INTERVENTION]\n"
            f"Audit Status: {ca_result['status']}\n"
            f"Gaps Detected: {violations}\n"
            "INSTRUCTION: " + " ".join(instructions) + "\n"
        )
        
        return intervention_prompt
