from typing import List, Dict

class IRQ:
    """
    Interactive Response Queue (IRQ)
    Translates raw System 2 (CAE) signals into System 1 (Model) Prompts.
    """
    def __init__(self):
        pass

    def translate_violations(
        self,
        violations: List[Dict],
        severity: int,
        gate_breakdown: List[Dict]
    ) -> str:
        """
        Input: [{'type': 'Math Error', 'detail': '2+2=5 is wrong'}]
        Output: "REFLECT: A calculation error was detected..."
        """
        if not violations:
            return (
                "REFLECT:\n"
                "- The draft appears structurally and logically sound.\n"
                "REVISE:\n"
                "Keep the response concise and correct."
            )

        prompt = "REFLECT:\n"
        for v in violations[:3]:
            prompt += f"- [{v['type']}]: {v['detail']}\n"

        if severity >= 2:
            prompt += "- The issues above are severe; fix them before adding new content.\n"

        if gate_breakdown:
            failed = [g["gate"] for g in gate_breakdown if not g["passed"]]
            if failed:
                prompt += f"- Gate failures: {', '.join(failed)}.\n"

        prompt += "REVISE:\nApply the fixes above and keep the answer focused."
        return prompt

    def create_injection_prompt(self, draft: str, reflection: str) -> str:
        """
        Combines Draft + Reflection to prompt for Revision.
        """
        return f"{draft}\n\n{reflection}\n\nREVISE:"

    def build_learned_rule(self, violations: List[Dict]) -> str:
        """
        Produce a single-sentence learned rule based on violations.
        """
        if not violations:
            return "Always answer directly and verify correctness before responding."

        type_map = {
            "Math Error": "Always verify arithmetic results before stating them.",
            "Logic Contradiction": "Avoid contradictions by keeping statements consistent.",
            "Grammar Error": "Write complete sentences with correct grammar and punctuation.",
            "Punctuation Spam": "Avoid punctuation spam; use normal sentence punctuation.",
            "Repetition": "Avoid repeating the same phrase or sentence.",
            "Boilerplate": "Do not include boilerplate sections like References or External links.",
            "Gibberish": "Use real words and clear sentences, not gibberish."
        }
        for v in violations:
            rule = type_map.get(v.get("type"))
            if rule:
                return rule

        return "Fix the highlighted issues before adding new content."
