import streamlit as st
import PyPDF2
import io

from preprocessing import preprocess
from similarity import match_percentage
from skill_extraction import compare_skills

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Skill Gap Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Skill Gap Analyzer")
st.markdown(
    "Upload your resume and paste a job description to see how well you match "
    "— and exactly which skills you need to add."
)
st.divider()

# ── Helper ────────────────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF file."""
    reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def score_color(score: float) -> str:
    if score >= 70:
        return "green"
    elif score >= 40:
        return "orange"
    return "red"


# ── Input columns ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📎 Your Resume")
    uploaded_resume = st.file_uploader(
        "Upload PDF resume", type=["pdf"], label_visibility="collapsed"
    )
    resume_text_raw = ""
    if uploaded_resume:
        resume_text_raw = extract_pdf_text(uploaded_resume)
        with st.expander("Preview extracted text"):
            st.text(resume_text_raw[:2000] + ("…" if len(resume_text_raw) > 2000 else ""))

with col2:
    st.subheader("📋 Job Description")
    jd_text_raw = st.text_area(
        "Paste the job description here",
        height=300,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning, SQL, and Tableau...",
        label_visibility="collapsed",
    )

st.divider()

# ── Analysis ──────────────────────────────────────────────────────────────────
analyze = st.button("🔍 Analyze Match", use_container_width=True, type="primary")

if analyze:
    if not uploaded_resume:
        st.warning("Please upload your resume PDF.")
        st.stop()
    if not jd_text_raw.strip():
        st.warning("Please paste the job description.")
        st.stop()

    with st.spinner("Analyzing your resume…"):
        # Preprocess
        resume_clean = preprocess(resume_text_raw)
        jd_clean = preprocess(jd_text_raw)

        # Cosine similarity score
        score = match_percentage(resume_clean, jd_clean)

        # Skill comparison
        skill_data = compare_skills(resume_text_raw, jd_text_raw)

    # ── Results ──────────────────────────────────────────────────────────────
    st.subheader("📊 Results")

    # Metric cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Match Score", f"{score}%")
    m2.metric("JD Skills Found", len(skill_data["jd_skills"]))
    m3.metric("✅ Matched Skills", len(skill_data["matched"]))
    m4.metric("❌ Missing Skills", len(skill_data["missing"]))

    # Score indicator
    color = score_color(score)
    st.markdown(
        f"<h3 style='color:{color}'>{'🟢 Strong match!' if score >= 70 else '🟡 Moderate match — bridge the gap.' if score >= 40 else '🔴 Low match — significant upskilling needed.'}</h3>",
        unsafe_allow_html=True,
    )

    st.progress(int(score))

    st.divider()

    # Skill breakdown
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### ✅ Matched Skills")
        if skill_data["matched"]:
            for skill in sorted(skill_data["matched"]):
                st.success(skill)
        else:
            st.info("No common skills detected.")

    with c2:
        st.markdown("#### ❌ Missing Skills")
        if skill_data["missing"]:
            for skill in sorted(skill_data["missing"]):
                st.error(skill)
        else:
            st.success("No missing skills — great fit!")

    with c3:
        st.markdown("#### 💡 Extra Skills (bonus)")
        if skill_data["extra"]:
            for skill in sorted(skill_data["extra"]):
                st.info(skill)
        else:
            st.write("—")

    st.divider()

    # Download missing skills as text
    if skill_data["missing"]:
        missing_text = "Skills to add to your resume:\n\n" + "\n".join(
            f"- {s}" for s in sorted(skill_data["missing"])
        )
        st.download_button(
            "⬇️ Download missing skills list",
            data=missing_text,
            file_name="missing_skills.txt",
            mime="text/plain",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · NLP powered by spaCy & scikit-learn · Author: Riddhi Garg (25BHI10004)")
