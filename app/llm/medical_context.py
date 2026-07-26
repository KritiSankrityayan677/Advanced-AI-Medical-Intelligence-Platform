"""
medical_context.py
Factual clinical context per tumor class.
Used to enrich reports and provide educational context in the UI.
All statements are general medical knowledge, not patient-specific claims.
"""

CLASS_CONTEXT = {
    "glioma": {
        "definition": (
            "Gliomas are tumors that arise from glial cells, the supportive tissue of the brain."
        ),
        "typical_location": (
            "They most commonly occur in the cerebral hemispheres, particularly the frontal "
            "and temporal lobes, though they can arise anywhere in the central nervous system."
        ),
        "clinical_note": (
            "Gliomas represent one of the most common primary brain tumors in adults. Their "
            "presentation and management vary significantly by grade and molecular subtype."
        ),
    },
    "meningioma": {
        "definition": (
            "Meningiomas are tumors that arise from the meninges — the membranes surrounding "
            "the brain and spinal cord."
        ),
        "typical_location": (
            "They most commonly occur along the cerebral convexities, parasagittal region, "
            "sphenoid wing, and skull base."
        ),
        "clinical_note": (
            "The majority of meningiomas are benign (WHO grade I) and often incidental findings. "
            "Small asymptomatic meningiomas are frequently managed with surveillance imaging."
        ),
    },
    "pituitary": {
        "definition": (
            "Pituitary tumors are lesions arising from the pituitary gland at the base of the brain."
        ),
        "typical_location": (
            "They are located within or above the sella turcica, the bony hollow at the base of the skull."
        ),
        "clinical_note": (
            "Most pituitary tumors are benign adenomas. They may be classified as functioning "
            "(hormone-secreting) or non-functioning, and management depends on size, symptoms, "
            "and hormonal activity."
        ),
    },
    "notumor": {
        "definition": (
            "No neoplastic finding was identified in the automated classification."
        ),
        "typical_location": (
            "The scan does not show features consistent with any of the tumor classes "
            "recognized by the model."
        ),
        "clinical_note": (
            "Absence of a classified tumor does not exclude other pathology. Correlation with "
            "clinical symptoms and specialist review remain important for a complete assessment."
        ),
    },
}


MODEL_PROVENANCE = (
    "The classification is produced by a deep learning model trained on thousands of "
    "annotated brain MRI scans. The highlighted region reflects the image areas that "
    "contributed most to this classification, based on patterns the model learned during training."
)


def get_context(predicted_class: str) -> dict:
    """Return the clinical context dictionary for a class, or a safe default."""
    return CLASS_CONTEXT.get(predicted_class, {
        "definition": "",
        "typical_location": "",
        "clinical_note": "",
    })