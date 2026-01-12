import json
import os
import re
from typing import Dict, Optional

import requests


class TruthTool:
    def __init__(self, enabled: bool = False, cache_path: Optional[str] = None, timeout_s: int = 6):
        self.enabled = enabled
        base_dir = os.path.join(os.path.dirname(__file__), "resources")
        os.makedirs(base_dir, exist_ok=True)
        self.cache_path = cache_path or os.path.join(base_dir, "truth_cache.json")
        self.timeout_s = timeout_s
        self.cache: Dict[str, str] = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except json.JSONDecodeError:
                self.cache = {}

    def verify_capital_claim(self, text: str) -> Optional[bool]:
        """
        Checks claims like "The capital of France is Paris."
        Returns True/False if verified, None if unverifiable or disabled.
        """
        if not self.enabled:
            return None

        pattern = r"capital of ([A-Za-z\s]+?) is ([A-Za-z\s]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None

        country = match.group(1).strip()
        claimed = match.group(2).strip()
        key = f"capital::{country.lower()}"
        if key in self.cache:
            return self._compare(self.cache[key], claimed)

        try:
            entity_id = self._lookup_entity(country)
            if not entity_id:
                return None
            capital = self._fetch_capital_label(entity_id)
        except requests.RequestException:
            return None

        if not capital:
            return None

        self.cache[key] = capital
        self._persist_cache()
        return self._compare(capital, claimed)

    def _lookup_entity(self, name: str) -> Optional[str]:
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        resp = requests.get("https://www.wikidata.org/w/api.php", params=params, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("search"):
            return None
        return data["search"][0].get("id")

    def _fetch_capital_label(self, entity_id: str) -> Optional[str]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        entity = data.get("entities", {}).get(entity_id, {})
        claims = entity.get("claims", {}).get("P36", [])
        if not claims:
            return None
        mainsnak = claims[0].get("mainsnak", {})
        value = mainsnak.get("datavalue", {}).get("value", {})
        capital_id = value.get("id")
        if not capital_id:
            return None
        return self._fetch_label(capital_id)

    def _fetch_label(self, entity_id: str) -> Optional[str]:
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        resp = requests.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        entity = data.get("entities", {}).get(entity_id, {})
        labels = entity.get("labels", {})
        label = labels.get("en", {}).get("value")
        return label

    @staticmethod
    def _compare(expected: str, claimed: str) -> bool:
        return expected.strip().lower() == claimed.strip().lower()

    def _persist_cache(self) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
