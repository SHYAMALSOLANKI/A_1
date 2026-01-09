import pytest
from pb2s_prompts import (
    get_core_policy,
    get_policy_profile,
    build_system_prompt
)

def test_get_core_policy():
    p = get_core_policy()
    assert "Immutable Core Policy" in p
    assert "Process-Based 2-System" in p

def test_get_policy_profile_loading():
    p1 = get_policy_profile("minimal")
    assert "Minimal" in p1
    p2 = get_policy_profile("logic_strict")
    assert "Logic Strict" in p2
    p3 = get_policy_profile("research")
    assert "Research" in p3

def test_get_policy_profile_fallback():
    p = get_policy_profile("non_existent_random_profile")
    # Should fallback to minimal
    assert "Profile: Minimal" in p

def test_build_system_prompt_composition():
    res = build_system_prompt("logic_strict", ["Rule: test rule"])
    assert "Immutable Core Policy" in res
    assert "Profile: Logic Strict" in res
    assert "Rule: test rule" in res
    assert "PPS" in res
