from typing import List, Dict

class IRQ:
    """
    Interactive Response Queue (IRQ)
    Translates raw System 2 (CAE) signals into System 1 (Model) Prompts.
    """
    def __init__(self):
        pass

    def translate_violations(self, violations: List[Dict]) -> str:
        """
        Input: [{'type': 'Math Error', 'detail': '2+2=5 is wrong'}]
        Output: "REFLECT: A calculation error was detected..."
        """
        if not violations:
            return "REFLECT: The draft appears structurally and logically sound. No immediate revisions logic detected."
        
        prompt = "REFLECT: The draft contains the following logic gaps:\n"
        for v in violations:
            prompt += f"- [{v['type']}]: {v['detail']}\n"
        
        prompt += "Correct these issues in the REVISE step accordingly."
        return prompt

    def create_injection_prompt(self, draft: str, reflection: str) -> str:
        """
        Combines Draft + Reflection to prompt for Revision.
        """
        return f"{draft}\n\n{reflection}\n\nREVISE:"
