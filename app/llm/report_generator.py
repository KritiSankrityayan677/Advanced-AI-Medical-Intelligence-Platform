"""
report_generator.py
Generates a structured preliminary medical report from a model prediction,
using a swappable LLM backend (currently Groq).
"""

from abc import ABC, abstractmethod
from typing import Dict

from groq import Groq

from app.core.config import LLM_PROVIDER, LLM_MODEL, GROQ_API_KEY
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt


# ── The abstract interface (the "socket") ──────────────
class ReportGenerator(ABC):
    """Abstract base class. Any LLM backend must implement generate()."""

    @abstractmethod
    def generate(self, predicted_class: str, confidence: float,
                 all_probabilities: Dict[str, float]) -> str:
        """Return the raw report text from the LLM."""
        ...


# ── The Groq implementation (one "plug") ───────────────
class GroqReportGenerator(ReportGenerator):
    """Generates reports using Groq's hosted open-source models."""

    def __init__(self, model: str = LLM_MODEL, api_key: str = GROQ_API_KEY):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example)."
            )
        self.model = model
        # Explicit timeout so a slow/unresponsive Groq call fails fast with a
        # clear error instead of hanging until Railway's edge kills the request.
        self.client = Groq(api_key=api_key, timeout=60.0, max_retries=1)

    def generate(self, predicted_class, confidence, all_probabilities) -> str:
        user_prompt = build_user_prompt(predicted_class, confidence, all_probabilities)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()


# ── Parse the raw text into structured sections ────────
def _parse_sections(text: str) -> Dict[str, str]:
    """Split the LLM output into its labelled sections."""
    sections = {
        "clinical_indication": "",
        "technique": "",
        "findings": "",
        "impression": "",
        "recommendations": "",
        "disclaimer": "",
    }
    labels = {
        "clinical indication:": "clinical_indication",
        "technique:": "technique",
        "findings:": "findings",
        "impression:": "impression",
        "recommendations:": "recommendations",
        "recommendation:": "recommendations",  # accept both spellings
        "disclaimer:": "disclaimer",
    }
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for label, key in labels.items():
            if stripped.lower().startswith(label):
                current = key
                sections[key] = stripped[len(label):].strip()
                matched = True
                break
        if not matched and current:
            sections[current] += " " + stripped
    return sections


# ── Cached generator instance ──────────────────────────
_generator_cache = None


def get_generator() -> ReportGenerator:
    """Return the configured report generator, creating it once."""
    global _generator_cache
    if _generator_cache is None:
        if LLM_PROVIDER == "groq":
            _generator_cache = GroqReportGenerator()
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    return _generator_cache


# ── The public function everything else calls ──────────
def generate_report(prediction: Dict) -> Dict:
    """
    Given a prediction dict (from predictor.predict), return a structured report.

    Returns:
        Dict with keys:
            raw_report: str — the full text from the LLM
            sections: Dict[str, str] — findings, impression, recommendation, disclaimer
            model: str — which model produced it
    """
    generator = get_generator()
    raw = generator.generate(
        prediction["predicted_class"],
        prediction["confidence"],
        prediction["all_probabilities"],
    )
    sections = _parse_sections(raw)

    # Safety net: never let a report go out without a disclaimer
    if not sections["disclaimer"]:
        sections["disclaimer"] = (
            "This report was produced by an automated system and does not constitute "
            "a medical diagnosis. It must be reviewed by a qualified radiologist "
            "before any clinical decision is made."
        )

    return {
        "raw_report": raw,
        "sections": sections,
        "model": LLM_MODEL,
    }


if __name__ == "__main__":
    import sys
    from app.models.predictor import predict

    if len(sys.argv) < 2:
        print("Usage: python -m app.llm.report_generator <path/to/image.jpg>")
        sys.exit(1)

    pred = predict(sys.argv[1])
    print(f"Prediction: {pred['predicted_class']} ({pred['confidence']:.1%})\n")

    result = generate_report(pred)
    print("----- GENERATED REPORT -----\n")
    print(result["raw_report"])
    print(f"\n(Generated by: {result['model']})")