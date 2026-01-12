import math
import re
from typing import Any, Dict, List, Tuple

import language_tool_python
import sympy
import textstat
from sentence_transformers import CrossEncoder

from services.stack_server.truth import TruthTool


class CAE:
    """
    Contradiction Audit Engine (CAE)
    Deterministic logic/math/grammar auditor (System 2).
    """
    def __init__(self, config: Dict[str, Any] | None = None, dqc_ref=None, truth_tool: TruthTool | None = None):
        self.config = config or {}
        self.dqc = dqc_ref
        self.truth_tool = truth_tool or TruthTool(enabled=False)

        self.enable_grammar = self.config.get("enable_grammar", True)
        self.enable_nli = self.config.get("enable_nli", False)

        self.grammar_tool = None
        if self.enable_grammar:
            self.grammar_tool = language_tool_python.LanguageTool("en-US")

        self.nli_model = None
        if self.enable_nli:
            self.nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base")

    def audit(self, text: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Main entry point. Audits text for logic, math, structure, and grammar.
        """
        metadata = metadata or {}
        difficulty = self._get_difficulty()

        hard_passed, hard_violations, severity, gate_breakdown = self.hard_gates(text, difficulty)
        verifier_violations = self.verifiers(text, metadata, difficulty)
        violations = hard_violations + verifier_violations
        severity = max([severity] + [v.get("severity", 1) for v in violations]) if violations else severity

        subscores: Dict[str, float] = {}
        if hard_passed:
            score, subscores = self.soft_score(text, metadata, difficulty, violations)
        else:
            score = 0.0

        passed = hard_passed and score >= self._pass_threshold()
        return {
            "passed": passed,
            "score": score,
            "violations": violations,
            "severity": severity,
            "gate_breakdown": gate_breakdown,
            "subscores": subscores,
        }

    def hard_gates(self, text: str, difficulty: float) -> Tuple[bool, List[Dict], int, List[Dict]]:
        violations: List[Dict] = []
        gate_breakdown: List[Dict] = []
        severity = 0

        text = text or ""
        if not text.strip():
            violations.append(self._violation("Empty Draft", "Draft is empty.", 3))
            gate_breakdown.append({"gate": "empty", "passed": False})
            return False, violations, 3, gate_breakdown
        gate_breakdown.append({"gate": "empty", "passed": True})

        punct_spam = self._is_punctuation_spam(text)
        gate_breakdown.append({"gate": "punctuation_spam", "passed": not punct_spam})
        if punct_spam:
            violations.append(self._violation("Punctuation Spam", "Punctuation spam detected.", 3))
            severity = max(severity, 3)

        boilerplate = self._has_wiki_boilerplate(text)
        gate_breakdown.append({"gate": "wiki_boilerplate", "passed": not boilerplate})
        if boilerplate:
            violations.append(self._violation("Boilerplate", "Wiki-style boilerplate detected.", 3))
            severity = max(severity, 3)

        trigram_loop = self._has_trigram_loop(text, difficulty)
        gate_breakdown.append({"gate": "trigram_loop", "passed": not trigram_loop})
        if trigram_loop:
            violations.append(self._violation("Repetition", "Trigram repetition loop detected.", 3))
            severity = max(severity, 3)

        if difficulty >= 0.6:
            valid_ratio = self._valid_word_ratio(text)
            gate_breakdown.append({"gate": "valid_word_ratio", "passed": valid_ratio >= 0.45})
            if valid_ratio < 0.45:
                violations.append(self._violation("Gibberish", "Low valid-word ratio.", 2))
                severity = max(severity, 2)
        else:
            gate_breakdown.append({"gate": "valid_word_ratio", "passed": True})

        if difficulty >= 0.8:
            diversity = self._unique_token_ratio(text)
            gate_breakdown.append({"gate": "diversity", "passed": diversity >= 0.4})
            if diversity < 0.4:
                violations.append(self._violation("Repetition", "Low diversity detected in output.", 2))
                severity = max(severity, 2)
        else:
            gate_breakdown.append({"gate": "diversity", "passed": True})

        passed = not any(v.get("severity", 1) >= 2 for v in violations)
        return passed, violations, severity, gate_breakdown

    def verifiers(self, text: str, metadata: Dict[str, Any], difficulty: float) -> List[Dict]:
        violations: List[Dict] = []
        violations.extend(self._check_math(text))
        violations.extend(self._check_contradictions(text, difficulty))

        truth_result = self.truth_tool.verify_capital_claim(text)
        if truth_result is False:
            violations.append(self._violation("Fact Error", "Capital claim does not match source of truth.", 2))

        return violations

    def soft_score(
        self,
        text: str,
        metadata: Dict[str, Any],
        difficulty: float,
        violations: List[Dict]
    ) -> Tuple[float, Dict[str, float]]:
        subscores: Dict[str, float] = {}

        grammar_score = 1.0
        if self.enable_grammar and self.grammar_tool:
            matches = self.grammar_tool.check(text)
            relevant_matches = [m for m in matches if m.ruleId not in ["UPPERCASE_SENTENCE_START", "WHITESPACE_RULE"]]
            grammar_score = max(0.0, 1.0 - (len(relevant_matches) * 0.1))
        subscores["grammar"] = grammar_score

        diversity = self._unique_token_ratio(text)
        subscores["diversity"] = diversity

        readability = 1.0
        try:
            grade = textstat.flesch_kincaid_grade(text)
            readability = max(0.0, 1.0 - max(0.0, (grade - 12) / 10))
        except Exception:
            readability = 0.8
        subscores["readability"] = readability

        base_score = (grammar_score * 0.4) + (diversity * 0.4) + (readability * 0.2)
        base_score = max(0.0, min(1.0, base_score))
        return base_score, subscores

    def _check_math(self, text: str) -> List[Dict]:
        violations: List[Dict] = []
        eq_pattern = r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
        matches = re.findall(eq_pattern, text)

        for m in matches:
            lhs_a, op, lhs_b, rhs = m
            expr_str = f"{lhs_a} {op} {lhs_b}"
            claimed_res = float(rhs)
            try:
                actual_res = float(sympy.sympify(expr_str))
                if abs(actual_res - claimed_res) > 0.01:
                    violations.append(self._violation(
                        "Math Error",
                        f"Calculation '{expr_str} = {rhs}' is incorrect. The truth is {actual_res}.",
                        2
                    ))
            except (sympy.SympifyError, ValueError):
                continue

        word_pattern = r"(?:result|sum|product) of ([\d\+\-\*\/\s]+?) is (\d+(?:\.\d+)?)"
        word_matches = re.findall(word_pattern, text, re.IGNORECASE)
        for expr, rhs in word_matches:
            expr = expr.strip()
            try:
                actual_res = float(sympy.sympify(expr))
                claimed_res = float(rhs)
                if abs(actual_res - claimed_res) > 0.01:
                    violations.append(self._violation(
                        "Math Error",
                        f"Statement '{expr} = {rhs}' is incorrect. The truth is {actual_res}.",
                        2
                    ))
            except (sympy.SympifyError, ValueError):
                continue

        return violations

    def _check_contradictions(self, text: str, difficulty: float) -> List[Dict]:
        violations: List[Dict] = []
        if not self.enable_nli or not self.nli_model:
            return violations

        sentences = [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 10]
        if len(sentences) < 2:
            return violations

        pairs = [(sentences[i], sentences[i + 1]) for i in range(min(len(sentences) - 1, 5))]
        scores = self.nli_model.predict(pairs)
        for idx, logits in enumerate(scores):
            probs = self._softmax(logits)
            contradiction_prob = probs[0]
            threshold = self._threshold_for_logic(difficulty)
            if contradiction_prob >= threshold:
                violations.append(self._violation(
                    "Logic Contradiction",
                    f"Sentences {idx + 1} and {idx + 2} appear contradictory.",
                    2
                ))
        return violations

    def _threshold_for_logic(self, difficulty: float) -> float:
        base = 0.6
        if self.dqc:
            base = self.dqc.get_threshold("logic")
        return min(0.9, max(0.5, base + difficulty * 0.1))

    def _pass_threshold(self) -> float:
        if self.dqc:
            return self.dqc.get_threshold("global")
        return 0.8

    def _get_difficulty(self) -> float:
        if self.dqc:
            return self.dqc.get_difficulty()
        return 0.3

    @staticmethod
    def _is_punctuation_spam(text: str) -> bool:
        punct = re.findall(r"[^\w\s]", text)
        if not punct:
            return False
        ratio = len(punct) / max(1, len(text))
        if ratio > 0.35:
            return True
        return bool(re.search(r"[.,!?]{6,}", text))

    @staticmethod
    def _has_wiki_boilerplate(text: str) -> bool:
        boilerplate = ["references", "external links", "other websites", "see also"]
        lower = text.lower()
        return any(b in lower for b in boilerplate)

    @staticmethod
    def _has_trigram_loop(text: str, difficulty: float) -> bool:
        words = re.findall(r"[A-Za-z']+", text.lower())
        if len(words) < 9:
            return False
        trigrams = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
        counts: Dict[str, int] = {}
        for tri in trigrams:
            counts[tri] = counts.get(tri, 0) + 1
        most_common = max(counts.values(), default=0)
        threshold = 4 if difficulty < 0.5 else 3
        return most_common >= threshold

    @staticmethod
    def _unique_token_ratio(text: str) -> float:
        tokens = re.findall(r"[A-Za-z']+", text.lower())
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _valid_word_ratio(self, text: str) -> float:
        words = re.findall(r"[A-Za-z']+", text)
        if not words:
            return 0.0
        valid = [w for w in words if self._is_valid_word(w)]
        return len(valid) / len(words)

    @staticmethod
    def _is_valid_word(word: str) -> bool:
        w = word.lower()
        if len(w) < 3 or len(w) > 20:
            return False
        if not w.isalpha():
            return False
        vowels = sum(1 for c in w if c in "aeiou")
        consonants = sum(1 for c in w if c.isalpha() and c not in "aeiou")
        if vowels == 0 or consonants == 0:
            return False
        unique_ratio = len(set(w)) / len(w)
        if unique_ratio < 0.3:
            return False
        entropy = CAE._shannon_entropy(w)
        return entropy >= 2.0

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = {}
        for c in text:
            counts[c] = counts.get(c, 0) + 1
        entropy = 0.0
        for count in counts.values():
            p = count / len(text)
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _softmax(logits) -> List[float]:
        max_logit = max(logits)
        exps = [math.exp(l - max_logit) for l in logits]
        denom = sum(exps)
        return [e / denom for e in exps]

    @staticmethod
    def _violation(vtype: str, detail: str, severity: int) -> Dict[str, Any]:
        return {"type": vtype, "detail": detail, "severity": severity}
