import re

class SafetyGuard:
    @staticmethod
    def check_injection(text: str) -> bool:
        """Simple checks for prompt injection."""
        patterns = [
            r"ignore previous instructions",
            r"system prompt:",
            r"you are now",
        ]
        for p in patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def redact_pii(text: str) -> str:
        """Naive redaction."""
        # Email
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL_REDACTED]', text)
        # Phone (simple)
        text = re.sub(r'\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}', '[PHONE_REDACTED]', text)
        return text

    @staticmethod
    def filter_bad_rule(rule_text: str) -> bool:
        """Returns True if rule is bad (blocked)."""
        blacklist = ["hack", "steal", "password", "ignore system"]
        for word in blacklist:
            if word in rule_text.lower():
                return True
        return False
