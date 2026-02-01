import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
from function import init_state, clear_cv, load_model

init_state()
model = load_model()

st.set_page_config(
    page_title="Resume Scanner Pro | ATS & AI Analysis",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Scanner Pro")

st.markdown("""
**Optimize your resume for both Applicant Tracking Systems (ATS) and human recruiters.**

This tool utilizes a **Dual-Engine Analysis** to score your CV against a Job Description:
* **🤖 Strict Mode (ATS Logic):** Uses TF-IDF to check for exact keyword matches. Essential for passing automated filters.
* **🧠 Smart Mode (AI Logic):** Uses Semantic Embeddings to understand the *context* and *relevance* of your skills, even if the wording differs.

---
""")

# Create a two-column layout for inputs
col_cv, col_description = st.columns(2)

# --- LEFT COLUMN: CV UPLOAD ---
with col_cv:
    uploaded_cv = st.file_uploader(
        label="📂 Upload Resume (PDF)",  # More descriptive label with emoji
        type=["pdf"],                    # Restrict to PDF only
        on_change=clear_cv,
        help="Please upload your CV/Resume in PDF format. Best results with 1-page documents." # User guidance
    )

# --- RIGHT COLUMN: JOB DESCRIPTION INPUT ---
with col_description:
    job_description = st.text_area(
        label="📋 Job Description",      # Clear label
        height=335,                      # Adjusted height to visually match the uploader
        placeholder="Paste the full job description here...", # Placeholder text for empty state
        help="Copy and paste the text from the job posting you want to apply for."
    )

# --- ACTION BUTTON ---
# Placed in the left column (under the uploader) for better UX
with col_cv:
    mode = st.selectbox(
        "Mode",
        ["Strict", "Flexibel"],
        index=0,
        help="",
        label_visibility="collapsed"
    )

    process = st.button(
        label="🔍 Analyze Match",        # Stronger "Call to Action" than just "Process"
        disabled=not (uploaded_cv and job_description), # Logic: Disable if either input is missing
        use_container_width=True,        # Make the button span the full column width
        help="Click to start the strict and semantic analysis."
    )

st.divider()

if uploaded_cv and job_description:
    if job_description.strip() == "":
        st.warning("KOSONG. TOLONG ISI DENGAN BENAR")
        st.stop()
    job_description_text = job_description

    if process and not st.session_state.info and not st.session_state.cv_text:
        reader = PyPDF2.PdfReader(uploaded_cv)
        all_pages = reader.pages

        total_pages = len(all_pages)
        if total_pages > 0:
            if total_pages > 1:
                st.session_state.info = f"File ada banyak, yaitu ada {total_pages}, tolong ..."
            elif total_pages == 1:
                st.session_state.info = f"bagusss, file anda hanya memiliki {total_pages}"

            for page in all_pages:
                st.session_state.cv_text += page.extract_text() + "\n\n"

        else:
            st.warning("KOSONG")
            st.stop()
    elif process and st.session_state.info and st.session_state.cv_text:

        st.toast("SUDAH ADA")

elif not uploaded_cv and not job_description:
    st.warning("GA ADA KEDUANYA")

elif uploaded_cv and job_description and not process:
    st.warning("TEKAN PROCESS DULU")
    
elif uploaded_cv or job_description:
    if uploaded_cv and not job_description:
        st.warning("GA ADA DESRIPSI")
        st.stop()
        
    elif not uploaded_cv and job_description:
        st.warning("GA ADA CV")

if st.session_state.cv_text and job_description_text \
    and st.session_state.info and mode.lower() == "strict":

    vectorizer = TfidfVectorizer()
    desc_vectors = vectorizer.fit_transform([job_description_text])
    cv_vectors = vectorizer.transform([st.session_state.cv_text])

    if st.session_state.info:
        st.info(st.session_state.info)

    similarity_scores = cosine_similarity(cv_vectors, desc_vectors)[0][0]
    st.metric("Nilai", value = similarity_scores)

    if similarity_scores > 0.5:
        st.success("LULUS")
    else:
        st.error("GAGAL, PERBAIKI")

elif st.session_state.cv_text and job_description_text \
    and st.session_state.info and mode.lower() == "flexibel":
    desc_embeds = model.encode(job_description_text)
    cv_embeds = model.encode(st.session_state.cv_text)

    if st.session_state.info:
        st.info(st.session_state.info)

    similarity_scores = util.cos_sim(cv_embeds, desc_embeds)
    st.metric("Nilai", value = similarity_scores)

    if similarity_scores > 0.5:
        st.success("LULUS")
    else:
        st.error("GAGAL, PERBAIKI")