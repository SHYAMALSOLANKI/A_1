from __future__ import annotations

import json
import os
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from memory import get_db  # your packages/memory should expose get_db in __init__.py
from memory.crud import (
    create_eval_run,
    create_full_trace,
    deactivate_rule,
    ensure_default_model_version,
    get_relevant_rules,
    list_rules,
)
from .providers import get_provider
from .safety import SafetyGuard
from .workflow import Pb2sWorkflow

app = FastAPI(
    title="PB2A Orchestrator",
    description="Non-commercial, self-hosted PB2S loop (Draft→Reflect→Revise→Learned).",
    version="0.2.0",
)

PROVIDER_TYPE = os.getenv("MODEL_PROVIDER", "mock")
PROVIDER_URL = os.getenv("MODEL_ENDPOINT", "http://localhost:8000")
DEFAULT_PROFILE = os.getenv("PB2A_POLICY_PROFILE", "minimal")

provider = get_provider(PROVIDER_TYPE, PROVIDER_URL)
workflow = Pb2sWorkflow(provider)


class ChatReq(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: Optional[str] = None
    policy_profile: Optional[str] = None  # User can override if allowed, or system set


class ChatResp(BaseModel):
    draft: str
    reflect_bullets: List[str]
    revision: str
    learned_rule: Optional[str]
    session_id: str
    turn_id: int
    eval: dict
    policy_profile: str


def _parse_session_id(session_id: Optional[str]) -> Optional[int]:
    if not session_id:
        return None
    try:
        return int(session_id)
    except ValueError:
        return None


def _basic_eval(draft: str, revision: str, bullets: List[str]) -> dict:
    """
    MVP eval gating:
    - reflect bullets <= 3 (already enforced)
    - revision non-empty
    - revision differs from draft if bullets exist
    """
    revision_ok = bool((revision or "").strip())
    improved = True
    if bullets:
        improved = (draft or "").strip() != (revision or "").strip()

    pass_fail = bool(revision_ok and improved)
    return {
        "revision_nonempty": revision_ok,
        "improved_if_criticized": improved,
        "reflect_bullets_count": len(bullets),
        "pass_fail": pass_fail,
    }


@app.post("/chat", response_model=ChatResp)
async def chat_endpoint(req: ChatReq, db=Depends(get_db)):
    if SafetyGuard.check_injection(req.message):
        raise HTTPException(status_code=400, detail="Safety rejection: prompt injection suspected.")

    # Determine profile
    profile = req.policy_profile or DEFAULT_PROFILE

    # retrieve rules
    rules = await get_relevant_rules(db, req.message, user_id=req.user_id, limit=3)
    rule_texts = [r.rule_text for r in rules]
    rule_ids = [r.id for r in rules]

    # run PB2S loop with profile
    result = await workflow.run_loop(req.message, rule_texts, profile_name=profile)

    safe_input = SafetyGuard.redact_pii(req.message)
    safe_draft = SafetyGuard.redact_pii(result["draft"])
    safe_revision = SafetyGuard.redact_pii(result["revision"])
    safe_reflect_raw = SafetyGuard.redact_pii(result["reflect_raw"])

    eval_obj = _basic_eval(safe_draft, safe_revision, result["reflect_bullets"])
    learned_to_store = result["learned_rule_candidate"] if eval_obj["pass_fail"] else None

    model_version_id = await ensure_default_model_version(db)

    turn = await create_full_trace(
        db,
        user_id=req.user_id,
        session_id=_parse_session_id(req.session_id),
        user_input_redacted=safe_input,
        draft_text=safe_draft,
        reflection_text=safe_reflect_raw,
        revision_text=safe_revision,
        model_version_id=model_version_id,
        retrieved_rule_ids=rule_ids,
        learned_rule_text=learned_to_store,
    )

    await create_eval_run(
        db,
        model_version_id=model_version_id,
        turn_id=turn.id,
        suite_name="basic",
        score=eval_obj,
        pass_fail=bool(eval_obj["pass_fail"]),
    )

    await db.commit()

    return ChatResp(
        draft=result["draft"],
        reflect_bullets=result["reflect_bullets"],
        revision=result["revision"],
        learned_rule=learned_to_store,
        session_id=str(turn.session_id),
        turn_id=turn.id,
        eval=eval_obj,
        policy_profile=result["policy_profile"]
    )


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/admin/rules")
async def admin_list_rules(db=Depends(get_db)):
    rules = await list_rules(db, limit=200)
    return [{"id": r.id, "text": r.rule_text, "active": r.is_active, "scope_user_id": r.scope_user_id} for r in rules]


@app.post("/admin/rules/{rule_id}/deactivate")
async def admin_deactivate_rule(rule_id: int, db=Depends(get_db)):
    await deactivate_rule(db, rule_id)
    await db.commit()
    return {"status": "deactivated", "id": rule_id}
