import pytest
import os
from unittest.mock import patch
from services.orchestrator.budget import get_budget, estimate_tokens

def test_estimate_tokens():
    assert estimate_tokens("hello") == 1
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10  # 40 chars / 4 = 10

@patch.dict(os.environ, {
    "PB2A_CONTEXT_TOKENS": "1000",
    "PB2A_DRAFT_RATIO": "0.1",
    "PB2A_REFLECT_RATIO": "0.2",
    "PB2A_REVISE_RATIO": "0.6",
    "PB2A_LEARNED_RATIO": "0.1"
})
def test_budget_exact_sum():
    b = get_budget()
    assert b.total_context == 1000
    assert b.draft_max == 100
    assert b.reflect_max == 200
    assert b.revise_max == 600
    assert b.learned_max == 100

@patch.dict(os.environ, {
    "PB2A_CONTEXT_TOKENS": "1000",
    "PB2A_DRAFT_RATIO": "0.5",
    "PB2A_REFLECT_RATIO": "0.5",
    "PB2A_REVISE_RATIO": "0.5",
    "PB2A_LEARNED_RATIO": "0.5"
})
def test_budget_normalization():
    # Sum is 2.0, so scaling factor is 0.5
    b = get_budget()
    assert b.total_context == 1000
    # 0.5 * 0.5 * 1000 = 250
    assert b.draft_max == 250
    assert b.revise_max == 250
