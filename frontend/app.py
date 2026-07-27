"""
Medical AI Platform — Streamlit frontend.
Run: streamlit run frontend/app.py
"""

import base64
import hashlib

import requests
import streamlit as st

from api_client import (
    check_health,
    predict_report,
    get_history,
    API_BASE_URL,
)

API_BASE_URL="https://neurovision-ai.up.railway.app"

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom styling for warmer look ─────────────────────
st.markdown(
    """
    <style>
    .disclaimer-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff4e6;
        border-left: 4px solid #ff9800;
        color: #5d4037;
        margin: 1rem 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar navigation ─────────────────────────────────
def sidebar():
    st.sidebar.title("🧠 Brain MRI Analysis")
    st.sidebar.markdown(
        "AI-assisted preliminary review of brain MRI scans."
    )
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Go to",
        ["Analyze a Scan", "Past Scans", "About"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ All outputs are preliminary and require radiologist review. "
        "This tool does not provide medical diagnoses."
    )

    # Silent health check — only warn if API is down
    try:
        check_health()
    except Exception:
        st.sidebar.error(
            "Service temporarily unavailable. Please try again shortly."
        )

    return page


# ── Analyze page ───────────────────────────────────────
def render_analyze_page():
    st.title("Analyze a Brain MRI Scan")
    st.markdown(
        "Upload a brain MRI image for automated classification. The system will "
        "produce a classification with region-of-interest visualization. A structured "
        "radiology report can be generated after review of the initial results."
    )

    uploaded = st.file_uploader(
        "Upload MRI scan",
        type=["jpg", "jpeg", "png"],
        help="Accepted file types: JPG, PNG",
    )

    if not uploaded:
        st.info("👆 Upload a scan to get started.")
        return

    image_bytes = uploaded.getvalue()
    upload_id = hashlib.md5(image_bytes).hexdigest()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image(image_bytes, caption="Your scan", use_container_width=True)
    with col_b:
        st.markdown("### Ready to analyze")
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {len(image_bytes) / 1024:.0f} KB")

        already_analyzed = st.session_state.get("last_upload_id") == upload_id

        if not already_analyzed:
            if st.button("🔬 Run classification", type="primary", use_container_width=True):
                _run_classification(image_bytes, uploaded.name, upload_id)
        else:
            st.success("Classification complete — see results below.")

    # Show results if analysis was already run for this upload
    if st.session_state.get("last_upload_id") == upload_id:
        _render_classification(
            st.session_state["last_result"],
            st.session_state["last_image"],
        )

        # Report generation section
        st.markdown("---")
        _render_report_section(upload_id, image_bytes, uploaded.name)


def _run_classification(image_bytes: bytes, filename: str, upload_id: str):
    """Call the API and store result; report is deferred."""
    with st.spinner("Analyzing scan..."):
        try:
            result = predict_report(image_bytes, filename=filename)
        except requests.exceptions.HTTPError as e:
            st.error(
                "We couldn't complete the analysis. "
                "Please check the file is a valid image and try again."
            )
            st.caption(f"(Technical details: {e.response.status_code})")
            return
        except requests.exceptions.RequestException:
            st.error(
                "The analysis service is not responding. "
                "Please try again in a few moments."
            )
            return

    st.session_state["last_upload_id"] = upload_id
    st.session_state["last_result"] = result
    st.session_state["last_image"] = image_bytes
    st.session_state["report_shown"] = False
    st.rerun()


def _render_classification(result: dict, image_bytes: bytes):
    """Render classification + XAI, without the LLM report."""
    st.markdown("---")

    friendly_class = _friendly_class_name(result["predicted_class"])
    context = _get_class_context(result["predicted_class"])

    # Summary metrics
    c1, c2 = st.columns(2)
    c1.metric("Classification", friendly_class)
    c2.metric("Model confidence", f"{result['confidence']:.0%}")

    if result["confidence"] < 0.70:
        st.warning(
            "Confidence for this classification is below 70%. "
            "Correlation with clinical findings is especially important."
        )

    # Class context (educational, based on general medical knowledge)
    if context.get("definition"):
        with st.expander(f"About {friendly_class.lower()}", expanded=True):
            st.markdown(f"**Definition.** {context['definition']}")
            st.markdown(f"**Typical location.** {context['typical_location']}")
            st.markdown(f"**Clinical note.** {context['clinical_note']}")

    # Probability distribution
    st.markdown("### Probability distribution")
    st.caption("Relative likelihood assigned to each class by the model.")
    st.bar_chart(
        {_friendly_class_name(k): v for k, v in result["all_probabilities"].items()}
    )

    # XAI section
    st.markdown("### Region of interest")
    st.markdown(
        "The classification model was trained on thousands of annotated brain MRI scans. "
        "During training, it learned visual patterns associated with each class. "
        "The overlay below highlights the image regions that most influenced this "
        "particular classification."
    )
    heatmap_bytes = base64.b64decode(result["heatmap_base64"])
    left, right = st.columns(2)
    with left:
        st.image(image_bytes, caption="Original scan", use_container_width=True)
    with right:
        st.image(heatmap_bytes, caption="Region of interest overlay", use_container_width=True)

    st.caption(
        f"Warmer colors (red/yellow) indicate higher influence on the classification. "
        f"{context.get('typical_location', '')}"
    )


def _render_report_section(upload_id: str, image_bytes: bytes, filename: str):
    """Section for generating and viewing the structured report."""
    result = st.session_state["last_result"]

    st.subheader("📄 Structured radiology report")

    if not st.session_state.get("report_shown"):
        st.markdown(
            "Generate a standard-format preliminary report for radiologist review. "
            "The report will include Clinical Indication, Technique, Findings, "
            "Impression, and Recommendations."
        )
        if st.button("📝 Generate report", type="primary"):
            st.session_state["report_shown"] = True
            st.rerun()
    else:
        _render_report(result)


def _render_report(result: dict):
    """Render the structured radiology report."""
    sections = result["report"]

    section_order = [
        ("clinical_indication", "Clinical indication"),
        ("technique", "Technique"),
        ("findings", "Findings"),
        ("impression", "Impression"),
        ("recommendations", "Recommendations"),
    ]

    for key, heading in section_order:
        content = sections.get(key, "").strip()
        if not content and key == "recommendations":
            content = sections.get("recommendation", "").strip()
        if content:
            st.markdown(f"**{heading}**")
            st.write(content)

    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ Important</strong><br>
            {sections['disclaimer']}<br><br>
            This report is <strong>not a diagnosis</strong>. It is intended to support review by a qualified radiologist, not to replace it.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Past scans page ────────────────────────────────────
def render_history_page():
    st.title("Past Scans")
    st.caption("Scans you've analyzed before, most recent first.")

    try:
        data = get_history(limit=50)
    except Exception:
        st.error("Could not load past scans. Please try again shortly.")
        return

    if data["total"] == 0:
        st.info("No scans yet. Head over to 'Analyze a Scan' to start.")
        return

    st.write(f"**{data['total']} scan(s)** in your history.")
    st.markdown("---")

    for record in data["predictions"]:
        friendly = _friendly_class_name(record["predicted_class"])
        ts = record["timestamp"][:10]

        with st.expander(f"{friendly} • {record['confidence']:.0%} confidence • {ts}"):
            st.write(f"**File:** {record['filename'] or '(unnamed)'}")
            st.write(f"**Analyzed on:** {record['timestamp'][:19].replace('T', ' at ')}")

            st.markdown("**How the AI weighed each possibility:**")
            st.bar_chart(
                {_friendly_class_name(k): v
                 for k, v in record["all_probabilities"].items()}
            )

            if record["report_text"]:
                st.markdown("**AI report:**")
                st.text(record["report_text"])


# ── About page ─────────────────────────────────────────
def render_about_page():
    st.title("About This Tool")

    st.markdown(
        """
        ### Purpose

        This platform provides automated preliminary analysis of brain MRI scans
        to support radiologist workflow. It produces a classification, a visual
        indication of the region that most influenced the classification, and a
        structured preliminary report in standard radiology format.

        ### Scope of analysis

        The system is trained to identify four categories on brain MRI:

        - **Glioma** — tumor of glial cell origin, commonly cerebral
        - **Meningioma** — tumor of the meninges, typically along the convexities or skull base
        - **Pituitary tumor** — lesion within the sella turcica
        - **No tumor detected** — no neoplastic finding identified

        ### How to interpret the output

        - **Classification** — the model's best assessment of the scan category.
        - **Model confidence** — a probability between 0 and 100%. Values below 70%
          should be interpreted with additional caution.
        - **Region of interest** — the highlighted region of the scan indicates
          where the classification was most strongly influenced. Warmer colors
          indicate higher influence.
        - **Preliminary report** — a structured draft in the standard radiology
          format (Clinical Indication, Technique, Findings, Impression, Recommendations).

        ### Important safety information

        This platform is a research and decision-support tool. It is not approved
        as a medical device. Outputs are preliminary, are **not a diagnosis**, and
        must always be reviewed and correlated with clinical findings by a qualified
        radiologist or physician before any clinical decision.
        """
    )


# ── Helpers ────────────────────────────────────────────
_FRIENDLY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor detected",
    "pituitary": "Pituitary tumor",
}


def _friendly_class_name(raw: str) -> str:
    return _FRIENDLY_NAMES.get(raw, raw.capitalize())

def _get_class_context(raw: str) -> dict:
    """Fetch clinical context for the predicted class."""
    try:
        from app.llm.medical_context import get_context
        return get_context(raw)
    except Exception:
        return {}


# ── Router ─────────────────────────────────────────────
def main():
    page = sidebar()
    if page == "Analyze a Scan":
        render_analyze_page()
    elif page == "Past Scans":
        render_history_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()