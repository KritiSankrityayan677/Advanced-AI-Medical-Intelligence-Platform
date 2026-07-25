"""
test_report_generator.py
Tests for the LLM report generator.
Logic tests run offline; the live generation test skips without an API key.
"""

import pytest

from app.core.config import GROQ_API_KEY
from app.llm.prompts import build_user_prompt, SYSTEM_PROMPT, CLASS_DESCRIPTIONS
from app.llm.report_generator import _parse_sections, generate_report


# ── Prompt logic (offline) ─────────────────────────────
def test_system_prompt_has_guardrails():
    """System prompt must contain the core safety guardrails."""
    lower = SYSTEM_PROMPT.lower()
    assert "not a doctor" in lower or "not provide a diagnosis" in lower
    assert "disclaimer" in lower


def test_class_descriptions_cover_all_classes():
    """Every known class must have a plain-language description."""
    for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
        assert cls in CLASS_DESCRIPTIONS


def test_user_prompt_includes_prediction():
    """The built prompt must contain the predicted class and confidence."""
    prompt = build_user_prompt("glioma", 0.90,
                               {"glioma": 0.90, "meningioma": 0.05,
                                "notumor": 0.03, "pituitary": 0.02})
    assert "glioma" in prompt
    assert "90" in prompt


# ── Parsing logic (offline) ────────────────────────────
def test_parse_sections_extracts_all_four():
    """Parser must split a well-formed report into four sections."""
    text = (
        "FINDINGS: A mass is observed.\n"
        "IMPRESSION: Consistent with glioma.\n"
        "RECOMMENDATION: Refer to neuro-oncology.\n"
        "DISCLAIMER: AI generated, not a diagnosis."
    )
    sections = _parse_sections(text)
    assert sections["findings"] == "A mass is observed."
    assert sections["impression"] == "Consistent with glioma."
    assert sections["recommendation"] == "Refer to neuro-oncology."
    assert sections["disclaimer"] == "AI generated, not a diagnosis."


def test_parse_sections_handles_multiline():
    """Parser must join multi-line section content."""
    text = (
        "FINDINGS: Line one.\nLine two.\n"
        "IMPRESSION: Ok.\nRECOMMENDATION: Ok.\nDISCLAIMER: Ok."
    )
    sections = _parse_sections(text)
    assert "Line one." in sections["findings"]
    assert "Line two." in sections["findings"]


# ── Live generation (needs API key + network) ──────────
@pytest.mark.skipif(not GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_generate_report_end_to_end():
    """Full report generation must return all four sections and a disclaimer."""
    prediction = {
        "predicted_class": "glioma",
        "confidence": 0.90,
        "all_probabilities": {
            "glioma": 0.90, "meningioma": 0.05,
            "notumor": 0.03, "pituitary": 0.02,
        },
    }
    result = generate_report(prediction)
    assert "raw_report" in result
    assert set(result["sections"].keys()) == {
        "findings", "impression", "recommendation", "disclaimer"
    }
    # Disclaimer must never be empty (safety net guarantees this)
    assert len(result["sections"]["disclaimer"]) > 0