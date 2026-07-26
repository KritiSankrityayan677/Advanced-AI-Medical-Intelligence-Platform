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
        "⚠️ This tool provides preliminary AI analysis only. "
        "All results must be reviewed by a qualified doctor."
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
        "Upload a brain MRI image below. Our AI will review the scan, "
        "highlight areas of interest, and prepare a preliminary report "
        "for your doctor to review."
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
    # Fingerprint the upload so we only analyze each file once per session
    upload_id = hashlib.md5(image_bytes).hexdigest()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image(image_bytes, caption="Your scan", use_container_width=True)
    with col_b:
        st.markdown("### Ready to review")
        st.write(f"**File:** {uploaded.name}")
        st.write(f"**Size:** {len(image_bytes) / 1024:.0f} KB")

        already_analyzed = st.session_state.get("last_upload_id") == upload_id

        if already_analyzed:
            st.success("This scan has already been analyzed below.")
        elif st.button("🔬 Analyze scan", type="primary", use_container_width=True):
            _run_analysis(image_bytes, uploaded.name, upload_id)


def _run_analysis(image_bytes: bytes, filename: str, upload_id: str):
    """Call the API once, store result in session_state to prevent double calls."""
    with st.spinner("Analyzing your scan — this can take up to a minute..."):
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

    # Record so re-runs don't retrigger the API
    st.session_state["last_upload_id"] = upload_id
    st.session_state["last_result"] = result
    st.session_state["last_image"] = image_bytes

    _render_results(result, image_bytes)


def _render_results(result: dict, image_bytes: bytes):
    st.success("Analysis complete.")
    st.markdown("---")

    friendly_class = _friendly_class_name(result["predicted_class"])

    # Prediction summary
    c1, c2 = st.columns(2)
    c1.metric("Finding", friendly_class)
    c2.metric("Confidence", f"{result['confidence']:.0%}")

    if result["confidence"] < 0.70:
        st.warning(
            "The AI is not very confident about this result. "
            "A doctor's review is especially important here."
        )

    # Class probabilities (renamed for clarity)
    st.markdown("### How the AI weighed each possibility")
    st.bar_chart(
        {_friendly_class_name(k): v for k, v in result["all_probabilities"].items()}
    )

    # Grad-CAM heatmap
    st.markdown("### What the AI focused on")
    st.caption(
        "The colored overlay shows the areas of the scan the AI paid most attention to. "
        "Warm colors (red/yellow) mean the AI looked there strongly."
    )
    heatmap_bytes = base64.b64decode(result["heatmap_base64"])
    left, right = st.columns(2)
    with left:
        st.image(image_bytes, caption="Original scan", use_container_width=True)
    with right:
        st.image(heatmap_bytes, caption="AI attention overlay", use_container_width=True)

    # Report
    st.markdown("### Preliminary AI report")

    for section, heading in [
        ("findings", "What the AI observed"),
        ("impression", "The AI's interpretation"),
        ("recommendation", "Suggested next steps"),
    ]:
        st.markdown(f"**{heading}**")
        st.write(result["report"][section])

    # Warm, human disclaimer
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ Important</strong><br>
            {result['report']['disclaimer']}<br><br>
            This is <strong>not a diagnosis</strong>. Always share these results with your doctor before making any medical decisions.
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
        ### What this is
        This is an AI-assisted tool that helps review brain MRI scans. Upload a scan,
        and the system will suggest what it sees, show you where it looked, and prepare
        a written summary that your doctor can review alongside the images.

        ### What it can suggest
        The AI has been trained to recognize four common findings on brain MRI scans:
        - Glioma
        - Meningioma
        - Pituitary tumor
        - No tumor detected

        ### How to use it well
        - Upload a clear MRI image (JPG or PNG).
        - Read the "confidence" number — if it's below 70%, the AI is unsure.
        - Look at the highlighted areas to see what caught the AI's attention.
        - Take everything to your doctor. Always.

        ### Important safety information
        This tool is intended to **support** medical review, not replace it. It has been
        built for educational and research purposes. It is not approved as a medical
        device. The results it produces are **not a diagnosis** and must always be
        interpreted by a qualified doctor or radiologist.

        If you or someone you love is worried about a scan or a symptom, please contact
        a healthcare provider directly.
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