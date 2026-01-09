from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import (
    Draft,
    EvalRun,
    LearnedRule,
    ModelVersion,
    Reflection,
    Revision,
    RuleLink,
    Session,
    Turn,
)


def stable_rule_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def get_or_create_session(
    db: AsyncSession, user_id: str, session_id: Optional[int]
) -> Session:
    if session_id is not None:
        res = await db.execute(
            select(Session).where(and_(Session.id == session_id, Session.user_id == user_id))
        )
        existing = res.scalar_one_or_none()
        if existing is not None:
            return existing

    sess = Session(user_id=user_id)
    db.add(sess)
    await db.flush()
    return sess


async def ensure_default_model_version(db: AsyncSession) -> str:
    """
    Ensures there is at least one active model version for the system.
    """
    res = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True))
    mv = res.scalar_one_or_none()
    if mv:
        return mv.id

    mv = ModelVersion(
        id="local-default",
        base_model="local/selfhosted",
        adapter_name=None,
        git_sha=None,
        is_active=True,
    )
    db.add(mv)
    await db.flush()
    return mv.id


async def get_relevant_rules(
    db: AsyncSession, query_text: str, user_id: Optional[str] = None, limit: int = 3
) -> List[LearnedRule]:
    """
    Keyword-only MVP retriever.
    Scopes: global (scope_user_id NULL) + per-user (scope_user_id=user_id).
    """
    filters = [LearnedRule.is_active == True]
    if user_id:
        filters.append(or_(LearnedRule.scope_user_id == None, LearnedRule.scope_user_id == user_id))
    else:
        filters.append(LearnedRule.scope_user_id == None)

    stmt = select(LearnedRule).where(and_(*filters))

    q = (query_text or "").strip()
    if q:
        words = [w for w in q.split() if w]
        if words:
            keyword_filters = [LearnedRule.rule_text.ilike(f"%{w}%") for w in words[:5]]
            stmt = stmt.where(or_(*keyword_filters))

    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def upsert_learned_rule(
    db: AsyncSession, rule_text: str, scope_user_id: Optional[str]
) -> LearnedRule:
    rh = stable_rule_hash(rule_text)

    res = await db.execute(select(LearnedRule).where(LearnedRule.rule_hash == rh))
    existing = res.scalar_one_or_none()
    if existing is not None:
        return existing

    rule = LearnedRule(rule_text=rule_text, rule_hash=rh, scope_user_id=scope_user_id)
    db.add(rule)
    await db.flush()
    return rule


async def deactivate_rule(db: AsyncSession, rule_id: int) -> None:
    await db.execute(
        update(LearnedRule).where(LearnedRule.id == rule_id).values(is_active=False)
    )


async def list_rules(db: AsyncSession, limit: int = 200) -> List[LearnedRule]:
    res = await db.execute(
        select(LearnedRule).order_by(LearnedRule.created_at.desc()).limit(limit)
    )
    return list(res.scalars().all())


async def create_eval_run(
    db: AsyncSession,
    model_version_id: str,
    turn_id: int,
    suite_name: str,
    score: dict,
    pass_fail: bool,
) -> EvalRun:
    row = EvalRun(
        model_version_id=model_version_id,
        turn_id=turn_id,
        suite_name=suite_name,
        score_json=json.dumps(score, ensure_ascii=False),
        pass_fail=pass_fail,
    )
    db.add(row)
    await db.flush()
    return row


async def create_full_trace(
    db: AsyncSession,
    *,
    user_id: str,
    session_id: Optional[int],
    user_input_redacted: str,
    draft_text: str,
    reflection_text: str,
    revision_text: str,
    model_version_id: str,
    retrieved_rule_ids: Optional[List[int]] = None,
    learned_rule_text: Optional[str] = None,
) -> Turn:
    """
    Creates/uses session, creates turn, stores draft/reflection/revision.
    Links retrieved rules and (optionally) a generated learned rule.
    """
    sess = await get_or_create_session(db, user_id=user_id, session_id=session_id)

    turn = Turn(session_id=sess.id, user_input=user_input_redacted)
    db.add(turn)
    await db.flush()

    # link retrieved rules
    for rid in (retrieved_rule_ids or []):
        db.add(RuleLink(turn_id=turn.id, learned_rule_id=rid, reason="retrieved"))

    draft = Draft(
        turn_id=turn.id,
        model_version_id=model_version_id,
        text=draft_text,
        tokens_in=0,
        tokens_out=0,
    )
    db.add(draft)
    await db.flush()

    refl = Reflection(draft_id=draft.id, text=reflection_text, severity=0)
    db.add(refl)

    rev = Revision(draft_id=draft.id, text=revision_text, tokens_out=0)
    db.add(rev)

    if learned_rule_text:
        rule = await upsert_learned_rule(db, learned_rule_text, scope_user_id=None)
        db.add(RuleLink(turn_id=turn.id, learned_rule_id=rule.id, reason="generated"))

    return turn
