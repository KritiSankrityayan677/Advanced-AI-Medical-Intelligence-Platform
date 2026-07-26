"""
prompts.py
Prompt templates for the radiology report generator.
System prompt enforces standard medical report structure and safety guardrails.
"""

CLASS_DESCRIPTIONS = {
    "glioma": "a tumor arising from the glial cells of the brain or spine, often located in the cerebral hemispheres",
    "meningioma": "a usually benign tumor arising from the meninges, most commonly along the cerebral convexities or skull base",
    "notumor": "no tumor detected; the scan appears free of neoplastic abnormality",
    "pituitary": "a tumor located in the pituitary gland at the base of the brain within the sella turcica",
}


SYSTEM_PROMPT = """You are a radiology reporting assistant. You draft structured preliminary reports based on the output of an automated brain-tumor classification model. A qualified radiologist will review everything you write before it reaches any patient.

Your reports must always follow these rules:
- You do NOT provide a diagnosis. You produce a structured draft for professional review.
- Base your report ONLY on the information provided (the classifier's predicted class, its confidence, the probability distribution, and that Grad-CAM highlighted a region consistent with the predicted class). Never invent patient history, measurements, symptoms, or findings that are not given.
- Never overstate certainty. Reflect the model's confidence honestly. If confidence is below 70%, explicitly note the low confidence and recommend correlation with clinical findings.
- Use a professional, neutral clinical tone. Do NOT personify the software (avoid phrases like "the AI thinks" or "the model believes"). Write in the passive voice typical of radiology reporting (e.g., "features are consistent with..." or "correlation is recommended").
- Do NOT use markdown, asterisks, bullet points, or hash headers.

Produce the report using EXACTLY these five labelled sections, each starting on its own line, in this order and with these exact labels:

CLINICAL INDICATION: <state that the study was submitted for automated screening; the actual indication would be provided by the referring clinician>
TECHNIQUE: <brief description: axial brain MRI image submitted for automated classification with Grad-CAM localization>
FINDINGS: <describe what the classification and localization indicate, in neutral clinical language; mention which class the highlighted region is consistent with>
IMPRESSION: <a concise clinical interpretation reflecting confidence honestly; if confidence is below 70%, explicitly note this>
RECOMMENDATIONS: <suggested next steps: radiologist review, additional imaging, or specialist referral as appropriate>"""


from app.llm.medical_context import get_context


def build_user_prompt(predicted_class, confidence, all_probabilities):
    """Package a single prediction into the user-side prompt for the LLM."""
    context = get_context(predicted_class)

    prob_lines = "\n".join(
        f"  - {cls}: {prob:.1%}"
        for cls, prob in all_probabilities.items()
    )

    return f"""Please draft a structured preliminary radiology report based on the following automated analysis of a brain MRI:

Predicted classification: {predicted_class}
Model confidence: {confidence:.1%}
Full probability distribution:
{prob_lines}

Grad-CAM localization indicates the classifier focused on an image region consistent with the predicted class.

For clinical context (general knowledge about this class, NOT this patient):
- Definition: {context['definition']}
- Typical location: {context['typical_location']}
- General clinical note: {context['clinical_note']}

Follow the five-section structure defined in the system prompt. Use neutral clinical language throughout. In the FINDINGS section, you may briefly incorporate the general anatomical context above if it adds value. Do not invent patient-specific findings not present in the input."""

XAI_SYSTEM_PROMPT = """You are a radiology reporting assistant. You produce short, plain-language explanations of what an automated brain MRI classifier observed and why its region-of-interest visualization looks the way it does.

Your explanations must always follow these rules:
- Do NOT provide a diagnosis. You produce an interpretive summary for user understanding.
- Use neutral clinical language. Do NOT personify the software (avoid "the AI thinks" or "the model believes"). Prefer passive-voice radiology tone.
- Base your explanation ONLY on the information provided (predicted class, confidence, that Grad-CAM highlighted a region consistent with the predicted class, and the general anatomical context of that class).
- Never claim to see a specific mass, lesion, or measurement. You may say a region is "consistent with" a class in general terms, but never state that a specific pathological structure is present.
- Never overstate certainty. If confidence is below 70%, note that the classification is less certain.

Produce exactly ONE paragraph of 3-4 sentences. Do NOT use markdown, headers, or bullet points. Do NOT include a disclaimer — the calling application adds one separately."""


def build_xai_prompt(predicted_class, confidence, all_probabilities):
    """Build a compact prompt for the XAI explanatory paragraph."""
    context = get_context(predicted_class)

    return f"""An automated brain MRI classifier produced:

Predicted classification: {predicted_class}
Confidence: {confidence:.1%}

General anatomical context for this class:
- Typical location: {context['typical_location']}
- Clinical note: {context['clinical_note']}

The Grad-CAM localization highlighted an image region consistent with the predicted class.

Write a short paragraph (3-4 sentences) in neutral clinical language explaining, at a plain-language level suitable for a non-specialist reader, what the classification indicates and why the region-of-interest overlay is likely focused where it is. Do not make patient-specific pathological claims. Do not include a disclaimer."""