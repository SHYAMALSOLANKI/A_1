import json
import os
import time
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self, path: str | None = None):
        base_dir = os.path.join(os.path.dirname(__file__), "resources")
        os.makedirs(base_dir, exist_ok=True)
        self.path = path or os.getenv(
            "STACK_MEMORY_PATH",
            os.path.join(base_dir, "memory.jsonl")
        )
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def add(self, learned: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        sentence = self._one_sentence(learned or "").strip()
        if not sentence:
            sentence = "Always respond clearly and verify correctness before replying."
        record = {
            "learned": sentence,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def query(self, prompt: str, top_k: int = 3) -> List[Dict[str, Any]]:
        prompt_tokens = set(self._tokenize(prompt))
        scored: List[Dict[str, Any]] = []
        for record in self._iter_records():
            learned = record.get("learned", "")
            learned_tokens = set(self._tokenize(learned))
            score = self._jaccard(prompt_tokens, learned_tokens)
            scored.append({"score": score, "learned": learned, "metadata": record.get("metadata", {})})

        scored.sort(key=lambda r: r["score"], reverse=True)
        results: List[Dict[str, Any]] = []
        token_budget = 200
        used_tokens = 0
        for item in scored:
            if len(results) >= top_k:
                break
            tokens = self._tokenize(item["learned"])
            if used_tokens + len(tokens) > token_budget:
                continue
            used_tokens += len(tokens)
            results.append(item)
        return results

    def _iter_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in "".join([c.lower() if c.isalnum() else " " for c in text]).split() if t]

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _one_sentence(text: str) -> str:
        for sep in [".", "!", "?"]:
            if sep in text:
                return text.split(sep)[0].strip() + sep
        return text.strip()
