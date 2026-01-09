from __future__ import annotations

import importlib.resources
import re
from pathlib import Path
from typing import Dict, List, Optional


SYSTEM_FILE = "PB2S_SYSTEM_prompt.txt"
PPS_FILE = "PB2S_PPS_v01.txt"
REFLECT_WRAP_FILE = "PB2S_REFLECT_wrap.txt"
IRQ_FILE = "PB2S_IRQ_Queue_Vo1.txt"
CORE_POLICY_FILE = "core_policy.txt"


def _load_resource(filename: str, pkg_dir: str = "") -> str:
    """
    Loads a text file from pb2s_prompts/resources.
    Tries relative path first (robust for editable/dev), then importlib (installed).
    """
    # 1. Try relative to this file
    resources_path = Path(__file__).parent / "resources"
    if pkg_dir:
        resources_path = resources_path / pkg_dir
    
    local_path = resources_path / filename
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    # 2. Fallback to standard package approach
    base = importlib.resources.files("pb2s_prompts") / "resources"
    if pkg_dir:
        base = base / pkg_dir
    
    target = base / filename
    if target.is_file():
        return target.read_text(encoding="utf-8")
        
    raise FileNotFoundError(f"Resource not found: {filename} in {pkg_dir}")


def get_core_policy() -> str:
    return _load_resource(CORE_POLICY_FILE)


def get_policy_profile(name: str) -> str:
    """
    Loads a policy profile from resources/policies/<name>.txt.
    Falls back to 'minimal' if not found.
    """
    safe_name = re.sub(r"[^a-z0-9_]", "", name.lower())
    try:
        return _load_resource(f"{safe_name}.txt", pkg_dir="policies")
    except FileNotFoundError:
        return _load_resource("minimal.txt", pkg_dir="policies")


def get_system_prompt() -> str:
    # Used for legacy verify, now redirects to build_system_prompt with default
    return build_system_prompt("minimal", [])


def get_pps_prompt() -> str:
    return _load_resource(PPS_FILE)


def get_reflect_wrap() -> str:
    return _load_resource(REFLECT_WRAP_FILE)


def get_irq_seed() -> str:
    return _load_resource(IRQ_FILE)


def build_system_prompt(profile_name: str, active_rules: List[str], extra: Optional[str] = None) -> str:
    """
    Composes the full system prompt:
    Core + Profile + PPS + Rules + Security/Extra
    """
    core = get_core_policy().strip()
    profile = get_policy_profile(profile_name).strip()
    pps = get_pps_prompt().strip()
    
    rules_text = "\n".join([f"- {r}" for r in active_rules]) if active_rules else "- (none)"
    
    prompt = (
        f"{core}\n\n"
        f"--- POLICY PROFILE: {profile_name.upper()} ---\n"
        f"{profile}\n\n"
        f"--- PPS INSTRUCTIONS ---\n"
        f"{pps}\n\n"
        f"--- ACTIVE LEARNED RULES ---\n"
        f"{rules_text}\n"
    )
    
    if extra:
        prompt += f"\n--- EXTRA INSTRUCTIONS ---\n{extra}\n"
        
    return prompt


_SECTION_RE = re.compile(r"(?im)^(DRAFT|REFLECT|REVISE|LEARNED)\s*:\s*", re.MULTILINE)


def split_sections(text: str) -> Dict[str, str]:
    """
    Split model output into PB2S sections based on headers:
      DRAFT:, REFLECT:, REVISE:, LEARNED:

    Returns dict with keys present in output. Values are stripped section bodies.
    """
    matches = list(_SECTION_RE.finditer(text or ""))
    out: Dict[str, str] = {}
    if not matches:
        return out

    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[name] = (text[start:end] or "").strip()
    return out


def validate_pb2s_sections(text: str) -> bool:
    """
    Requires all 4 headers to be present in strict order.
    Does NOT require that content is non-empty (though callers usually should).
    """
    sections = list(_SECTION_RE.finditer(text or ""))
    names = [m.group(1).upper() for m in sections]
    required = ["DRAFT", "REFLECT", "REVISE", "LEARNED"]
    if any(r not in names for r in required):
        return False

    # strict order: first occurrence positions must be increasing in that order
    pos = []
    for r in required:
        for m in sections:
            if m.group(1).upper() == r:
                pos.append(m.start())
                break
    return pos == sorted(pos)


def parse_reflect_bullets(reflect_text: str) -> List[str]:
    """
    Parse bullets from REFLECT section; enforce <= 3 by truncating.
    Accepts '-', '*', '1.' styles.
    """
    bullet_pattern = r"^\s*?(-|\*|\d+\.)\s+(.+)$"
    matches = re.findall(bullet_pattern, reflect_text or "", re.MULTILINE)
    bullets = [m[1].strip() for m in matches if m[1].strip()]
    return bullets[:3]  # ENFORCE ≤3


def extract_learned_rule(text: str) -> Optional[str]:
    """
    Extract LEARNED rule from either:
    - a LEARNED section, or
    - a line like 'LEARNED: <rule>'
    Enforces max length (<=200 chars) and rejects 'None'.
    """
    sections = split_sections(text or "")
    candidate = None

    if "LEARNED" in sections:
        candidate = sections["LEARNED"].strip()
    else:
        m = re.search(r"(?im)^LEARNED\s*:\s*(.+)$", text or "")
        if m:
            candidate = m.group(1).strip()

    if not candidate:
        return None

    if candidate.lower() == "none":
        return None

    candidate = re.sub(r"\s+", " ", candidate).strip()
    if len(candidate) > 200:
        return None
    if len(candidate) < 5:
        return None

    return candidate


def render_reflect_prompt(draft_text: str) -> str:
    """
    Build reflect prompt using the PB2S reflect wrapper + injected draft.
    Wrapper already specifies a strict format.
    """
    wrap = get_reflect_wrap().strip()
    return f"{wrap}\n\nDRAFT:\n<<<{draft_text}>>>"
