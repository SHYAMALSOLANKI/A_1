import pytest
import re
from pb2s_prompts import (
    get_system_prompt,
    get_pps_prompt,
    render_reflect_prompt,
    validate_pb2s_sections,
    parse_reflect_bullets
)

def test_get_system_prompt():
    prompt = get_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0

def test_get_pps_prompt():
    prompt = get_pps_prompt()
    assert isinstance(prompt, str)
    # PPS content might vary, but should check basic load
    assert len(prompt) > 0

def test_render_reflect_prompt():
    draft = "This is a draft solution."
    prompt = render_reflect_prompt(draft)
    assert draft in prompt
    # The wrap usually mentions "DRAFT:"
    assert "DRAFT:" in prompt

def test_validate_pb2s_sections_success():
    valid_text = """
DRAFT:
This is the draft.

REFLECT:
- Check 1
- Check 2

REVISE:
Revised text.

LEARNED:
Core principle.
"""
    assert validate_pb2s_sections(valid_text) is True

def test_validate_pb2s_sections_missing_header():
    invalid_text = """
DRAFT:
...
REVISE:
...
LEARNED:
...
"""
    # Missing REFLECT
    assert validate_pb2s_sections(invalid_text) is False

def test_validate_pb2s_sections_wrong_order():
    invalid_text = """
REFLECT:
...
DRAFT:
...
REVISE:
...
LEARNED:
...
"""
    assert validate_pb2s_sections(invalid_text) is False

def test_parse_reflect_bullets_truncation():
    # Enforce <= 3
    reflect_text = """
- Bullet 1
- Bullet 2
- Bullet 3
- Bullet 4
"""
    bullets = parse_reflect_bullets(reflect_text)
    assert len(bullets) == 3
    assert bullets[0] == "Bullet 1"
    assert bullets[2] == "Bullet 3"

def test_parse_reflect_bullets_styles():
    text = """
* Star style
1. Number style
- Dash style
"""
    bullets = parse_reflect_bullets(text)
    assert len(bullets) == 3
    assert "Star style" in bullets
    assert "Number style" in bullets
    assert "Dash style" in bullets
