import json
import os
import time
from typing import Any, Dict


class TraceStore:
    def __init__(self, path: str | None = None):
        base_dir = os.path.join(os.path.dirname(__file__), "resources")
        os.makedirs(base_dir, exist_ok=True)
        self.path = path or os.getenv(
            "STACK_TRACE_PATH",
            os.path.join(base_dir, "traces.jsonl")
        )
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                f.write("")

    def append(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record["timestamp"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
