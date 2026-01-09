import json
import os
from typing import Dict, List

class PPS:
    """
    Persistent Prompt Stack (PPS)
    Responsible for loading and serving the 'Constitutional' rules of the system.
    """
    def __init__(self, policy_path: str = "resources/core_policy.json"):
        # Adjust path relative to where execution happens if needed, 
        # or use absolute paths in production config.
        # Docker Environment: /app/resources/core_policy.json
        if not os.path.exists(policy_path):
             # Try absolute path for Docker
             policy_path = "/app/resources/core_policy.json"
        
        self.policy_path = policy_path
        self.policies = self._load_policies()

    def _load_policies(self) -> Dict:
        try:
            with open(self.policy_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"PPS Error: Policy file not found at {self.policy_path}")
            return {}

    def get_system_prompt(self) -> str:
        """
        Compiles the active core rules into a System Prompt.
        """
        identity = self.policies.get("identity", {})
        logic = self.policies.get("logic", {})
        
        prompt = "SYSTEM CONSTITUTION:\n"
        for k, v in identity.items():
            prompt += f"- {v}\n"
        for k, v in logic.items():
            prompt += f"- {v}\n"
        
        return prompt

    def get_audit_checklist(self) -> List[str]:
        """
        Returns a list of rules the CAE should check against.
        """
        logic = self.policies.get("logic", {}).values()
        safety = self.policies.get("safety", {}).values()
        return list(logic) + list(safety)
