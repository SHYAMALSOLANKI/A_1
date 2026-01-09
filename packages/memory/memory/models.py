from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .db import Base


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Turn(Base):
    __tablename__ = "turns"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    user_input = Column(Text, nullable=False)  # store REDACTED input
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    session = relationship("Session", backref="turns")
    draft = relationship("Draft", uselist=False, back_populates="turn")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(String, primary_key=True)  # e.g. "local-default"
    base_model = Column(String, nullable=False)
    adapter_name = Column(String, nullable=True)
    git_sha = Column(String, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Draft(Base):
    __tablename__ = "drafts"
    id = Column(Integer, primary_key=True, index=True)
    turn_id = Column(Integer, ForeignKey("turns.id"), unique=True, nullable=False)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)

    text = Column(Text, nullable=False)
    tokens_in = Column(Integer, default=0)
    tokens_out = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    turn = relationship("Turn", back_populates="draft")
    reflection = relationship("Reflection", uselist=False, back_populates="draft")
    revision = relationship("Revision", uselist=False, back_populates="draft")


class Reflection(Base):
    __tablename__ = "reflections"
    id = Column(Integer, primary_key=True, index=True)
    draft_id = Column(Integer, ForeignKey("drafts.id"), unique=True, nullable=False)

    text = Column(Text, nullable=False)
    severity = Column(Integer, default=0)  # 0=info, 1=warning, 2=error
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    draft = relationship("Draft", back_populates="reflection")


class Revision(Base):
    __tablename__ = "revisions"
    id = Column(Integer, primary_key=True, index=True)
    draft_id = Column(Integer, ForeignKey("drafts.id"), unique=True, nullable=False)

    text = Column(Text, nullable=False)
    tokens_out = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    draft = relationship("Draft", back_populates="revision")


class LearnedRule(Base):
    __tablename__ = "learned_rules"
    id = Column(Integer, primary_key=True, index=True)
    rule_text = Column(Text, nullable=False)
    rule_hash = Column(String, index=True, unique=True, nullable=False)
    is_active = Column(Boolean, default=True)
    scope_user_id = Column(String, nullable=True)  # NULL = global
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RuleLink(Base):
    __tablename__ = "rule_links"
    id = Column(Integer, primary_key=True)
    turn_id = Column(Integer, ForeignKey("turns.id"), nullable=False)
    learned_rule_id = Column(Integer, ForeignKey("learned_rules.id"), nullable=False)
    reason = Column(String, nullable=True)  # "retrieved" or "generated"


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(Integer, primary_key=True, index=True)
    model_version_id = Column(String, ForeignKey("model_versions.id"), nullable=False)
    turn_id = Column(Integer, ForeignKey("turns.id"), nullable=False)

    suite_name = Column(String, nullable=False)
    score_json = Column(Text, nullable=False)  # JSON string
    pass_fail = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
