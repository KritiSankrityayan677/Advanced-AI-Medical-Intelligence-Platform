"""
prompts.py
Prompt templates for the LLM medical report generator.
The system prompt enforces safety guardrails and output structure.
"""

# Plain-language descriptions to ground the model in what each class means
CLASS_DESCRIPTIONS = {
    "glioma": "a tumor arising from the glial cells of the brain or spine",
    "meningioma": "a usually benign tumor arising from the meninges, the membranes surrounding the brain and spinal cord",
    "notumor": "no tumor detected; the scan appears free of neoplastic abnormality",
    "pituitary": "a tumor located in the pituitary gland at the base of the brain",
}


SYSTEM_PROMPT = """You are a medical imaging report assistant. You draft structured, PRELIMINARY reports based on the output of an automated brain-tumor classification model. A qualified radiologist will review everything you write.

You must always follow these rules:
- You are NOT a doctor and you do NOT provide a diagnosis. Everything you produce is an AI-generated draft for professional review.
- Base your report ONLY on the information provided (the model's predicted class, its confidence, and that Grad-CAM highlighted the region consistent with that class). Never invent patient history, measurements, symptoms, or findings that are not given.
- Never overstate certainty. Reflect the model's confidence honestly. If confidence is below 70%, explicitly state that the result is low-confidence and should be interpreted with caution.
- Always end with a disclaimer that this is an automated aid, not a medical diagnosis.
- Use clear, professional language. Do NOT use markdown, asterisks, bullet points, or hash headers.

Produce the report using EXACTLY these four labelled sections, each starting on its own line, in this order and with these exact labels:
FINDINGS: <what the automated analysis observed>
IMPRESSION: <a concise interpretation reflecting the confidence honestly>
RECOMMENDATION: <suggested next step, e.g. specialist review or further imaging>
DISCLAIMER: <a clear statement that this is AI-generated and not a diagnosis>"""


def build_user_prompt(predicted_class, confidence, all_probabilities):
    """
    Package a single prediction into the user-side prompt for the LLM.
    """
    description = CLASS_DESCRIPTIONS.get(predicted_class, "an unspecified finding")

    prob_lines = "\n".join(
        f"  - {cls}: {prob:.1%}"
        for cls, prob in all_probabilities.items()
    )

    return f"""An automated brain MRI classifier analysed a single scan and produced:

Predicted class: {predicted_class} ({description})
Confidence in the predicted class: {confidence:.1%}
Full probability distribution across all classes:
{prob_lines}

The Grad-CAM explainability method indicates the model focused on the image region most consistent with the predicted class.

Write a preliminary structured report for radiologist review, based ONLY on the information above."""