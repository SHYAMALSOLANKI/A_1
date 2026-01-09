from __future__ import annotations

from typing import Any, Dict, List, Optional

from pb2s_prompts import (
    build_system_prompt,
    extract_learned_rule,
    get_pps_prompt,
    get_system_prompt,
    parse_reflect_bullets,
    render_reflect_prompt,
    split_sections,
)

from .budget import get_budget, estimate_tokens
from .providers import ModelProvider
from .safety import SafetyGuard


class Pb2sWorkflow:
    def __init__(self, provider: ModelProvider):
        self.provider = provider
        self.budget = get_budget()

    async def run_loop(self, user_input: str, active_rules: List[str], profile_name: str = "minimal") -> Dict[str, Any]:
        """
        Efficient PB2S loop utilizing configured policy profile and token budgeting.
        """
        # 1. Build System Prompt via Prompt Builder
        full_system = build_system_prompt(profile_name, active_rules)

        # 2. DRAFT Step
        draft_prompt = (
            "Output MUST start with:\n"
            "DRAFT:\n"
            "Then the draft answer.\n\n"
            f"USER_INPUT:\n<<<{user_input}>>>"
        )
        # Use budget
        draft_raw = await self.provider.generate(
            draft_prompt, 
            system=full_system,
            max_tokens=self.budget.draft_max
        )
        
        # Parse
        draft_sections = split_sections(draft_raw)
        draft_text = draft_sections.get("DRAFT", draft_raw).strip()

        # 3. REFLECT + REVISE Step
        # The prompt wrapper for reflection
        reflect_prompt = render_reflect_prompt(draft_text)
        
        reflect_raw = await self.provider.generate(
            reflect_prompt, 
            system=full_system, 
            max_tokens=self.budget.reflect_max
        )
        
        sections = split_sections(reflect_raw)
        reflect_text = sections.get("REFLECT", reflect_raw).strip()
        bullets = parse_reflect_bullets(reflect_text)  # Truncates to <= 3

        # 4. REVISE Step
        # If REVISE was not in output, force it
        revision_text = sections.get("REVISE", "").strip()
        learned_text_raw = sections.get("LEARNED", "").strip()
        
        if not revision_text:
            revise_prompt = (
                "Output MUST start with:\nREVISE:\n\n"
                f"USER_INPUT:\n<<<{user_input}>>>\n\n"
                f"DRAFT:\n<<<{draft_text}>>>\n\n"
                f"REFLECT_BULLETS:\n" + "\n".join([f"- {b}" for b in bullets]) + "\n\n"
                "Then provide the REVISED response.\n"
                "Finally, provide LEARNED rule or 'LEARNED: None'."
            )
            rev_raw = await self.provider.generate(revise_prompt, system=full_system)
            sections_rev = split_sections(rev_raw)
            revision_text = sections_rev.get("REVISE", rev_raw).strip()
            if "LEARNED" in sections_rev:
                learned_text_raw = sections_rev["LEARNED"]

        # 5. LEARNED Extraction
        learned_candidate = extract_learned_rule(learned_text_raw or reflect_raw or "")
        
        # Safety Check on Rule
        if learned_candidate:
            if SafetyGuard.filter_bad_rule(learned_candidate):
                learned_candidate = None
            elif len(learned_candidate) > 200:
                learned_candidate = None

        return {
            "draft": draft_text,
            "reflect_raw": reflect_raw,  # contains full output usually
            "reflect_bullets": bullets,
            "revision": revision_text,
            "learned_rule_candidate": learned_candidate,
            "policy_profile": profile_name
        }
