import streamlit as st
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import util
# Import custom helper functions to keep the main code clean
from function import init_state, clear_cv, load_model

# --- INITIALIZATION ---
# Initialize session state variables (like 'cv_text' and 'info') to prevent KeyErrors on startup
init_state()

# Load the SBERT model. This function is cached (@st.cache_resource) so the model 
# only loads once, ensuring the app runs fast after the first launch.
model = load_model()

# --- PAGE CONFIGURATION ---
# Set up the browser tab properties and layout
st.set_page_config(
    page_title="Resume Scanner Pro | ATS & AI Analysis",
    page_icon="📄",
    layout="wide"  # 'wide' layout provides more horizontal space for the two-column design
)

# --- HEADER SECTION ---
st.title("📄 Resume Scanner Pro")

# Display the introductory text and explain the "Dual-Engine" concept to the user
st.markdown("""
**Optimize your resume for both Applicant Tracking Systems (ATS) and human recruiters.**

This tool utilizes a **Dual-Engine Analysis** to score your CV against a Job Description:
* **🤖 Strict Mode (ATS Logic):** Uses TF-IDF to check for exact keyword matches. Essential for passing automated filters.
* **🧠 Smart Mode (AI Logic):** Uses Semantic Embeddings to understand the *context* and *relevance* of your skills, even if the wording differs.

---
""")

# --- UI LAYOUT CONFIGURATION ---
# Create a balanced two-column layout to separate inputs side-by-side.
# This ensures the Resume and Job Description are visible simultaneously.
col_cv, col_description = st.columns(2)

# --- LEFT COLUMN: CV UPLOAD ---
with col_cv:
    uploaded_cv = st.file_uploader(
        label="📂 Upload Resume (PDF)",   # User-friendly label with visual indicator
        type=["pdf"],                     # Enforce PDF format to ensure text extraction compatibility
        on_change=clear_cv,               # Callback: Resets previous analysis results when a new file is uploaded
        help="Please upload your CV/Resume in PDF format. Best results with 1-page documents."
    )

# --- RIGHT COLUMN: JOB DESCRIPTION INPUT ---
with col_description:
    job_description = st.text_area(
        label="📋 Job Description",
        # Height is manually set to 335px to align visually with the file uploader component
        height=335,                       
        placeholder="Paste the full job description here...",
        help="Copy and paste the text from the job posting you want to apply for."
    )

# --- ANALYSIS CONTROLS ---
# Placing the controls in the left column (under the uploader) provides a natural top-to-bottom flow.
with col_cv:
    # Dropdown to select the analysis algorithm (Statistical vs. Semantic)
    mode = st.selectbox(
        "Mode",
        ["Strict", "Flexible"],
        index=0,
        help="Select 'Strict' for keyword matching or 'Flexible' for context matching.",
        label_visibility="collapsed"      # Hides the label for a cleaner UI look
    )

    # Primary Action Button
    process = st.button(
        label="🔍 Analyze Match",
        # Conditional Logic: The button remains disabled/unclickable until both inputs are provided.
        # This prevents the user from triggering an analysis with incomplete data.
        disabled=not (uploaded_cv and job_description), 
        use_container_width=True,         # Expands the button to fill the column width
        help="Click to start the strict or semantic analysis."
    )

# Visual separator between inputs and results
st.divider()

# --- LOGIC & VALIDATION ---
if uploaded_cv and job_description:
    # 1. INPUT VALIDATION: Ensure the Job Description is not empty/whitespace only
    if job_description.strip() == "":
        st.warning("⚠️ Job Description cannot be empty. Please paste the text to proceed.")
        st.stop()

    # 2. CORE PROCESSING: Execute only if the 'Analyze' button is clicked AND data is not yet cached
    if process and not st.session_state.info and not st.session_state.cv_text:
        try:
            # Attempt to read and parse the uploaded PDF file
            reader = PyPDF2.PdfReader(uploaded_cv)
            all_pages = reader.pages

            total_pages = len(all_pages)
            
            # Check if PDF content exists
            if total_pages > 0:
                if total_pages > 1:
                    # Logic: Provide specific feedback based on resume length
                    # Note: Warns about potential visibility issues with multiple pages
                    st.session_state.info = f"ℹ️ **Note ({total_pages} Pages Detected):** Recruiters often prioritize the first page. Ensure your most critical skills and experience are listed on Page 1 to avoid being overlooked."
                elif total_pages == 1:
                    # Affirmation: Confirms that the length meets industry standards
                    st.session_state.info = f"✅ **Optimal Length:** Single-page resume detected. This concise format is highly preferred by recruiters and ATS for quick scanning."

                # Extraction Loop: Iterate through all pages and append text to session state
                for page in all_pages:
                    st.session_state.cv_text += page.extract_text() + "\n\n"
            else:
                # Handle cases where PDF is valid but empty
                st.warning("⚠️ Error: The uploaded PDF appears to be empty or unreadable.")
                st.stop()
                
        except Exception as e:
            # Graceful Error Handling: Catch unexpected errors (e.g., encrypted files)
            error_msg = str(e).lower()
            answer = "" # Placeholder for the error message

            # 1. Handle Password Protected / Encrypted PDFs
            if "password" in error_msg or "encrypted" in error_msg:
                answer = "🔒 **Encrypted PDF Detected**\n\nThis file is password protected. Please upload an **unlocked (decrypted)** version of your resume so the system can read it."

            # 2. Handle Corrupt Files or Invalid Formats (e.g., renaming .docx to .pdf manually)
            elif "pdf marker" in error_msg or "eof" in error_msg or "startxref" in error_msg:
                answer = "❌ **Invalid or Corrupt File**\n\nThe file appears to be corrupted or is not a valid PDF format. \n\n*Tip: If you renamed a Word file (.docx) to .pdf, please open it in Word and choose 'Save as PDF' instead.*"

            # 3. Handle Empty Files (Zero bytes)
            elif "empty" in error_msg or "no data" in error_msg:
                answer = "⚠️ **Empty File**\n\nThe uploaded file appears to contain no data. Please check the file and try again."

            # 4. Handle General/Unknown Errors
            else:
                answer = f"⚠️ **Processing Error**\n\nAn unexpected error occurred while reading the PDF.\n\n**Technical Details:** `{str(e)}`"

            # Display the specific error message to the user
            st.error(answer)
            st.stop()
            
    # 3. CACHING STRATEGY: If analysis data exists, use cache instead of re-processing
    elif process and st.session_state.info and st.session_state.cv_text:
        st.toast("✅ Using cached resume data.")
        
    # 4. READY STATE: Inputs are valid, waiting for user action
    elif not process and not st.session_state.info and not st.session_state.cv_text:
        st.warning("⚠️ Ready to analyze. Please click the 'Analyze Match' button to start.")  

# --- USER ONBOARDING & INSTRUCTIONS ---
# Scenario 1: Initial State (No inputs provided)
# Display a clear "How-to" guide to assist first-time users
elif not uploaded_cv and not job_description:
    st.info("""
    👋 **Welcome! To start the analysis, please follow these steps:**
    1. Upload your **Resume (PDF)** on the left panel.
    2. Paste the **Job Description** on the right panel.
    3. Select your desired **Mode** (Strict or Flexible).
    4. Click the **'Analyze Match'** button.
    """)
    
# Scenario 2: Partial Input Handling
# Provide specific feedback on exactly which component is missing (Resume vs. JD)
elif uploaded_cv or job_description:
    if uploaded_cv and not job_description:
        st.warning("⚠️ **Incomplete Input:** You have uploaded a **Resume**, but the **Job Description** is missing.")
        
    elif not uploaded_cv and job_description:
        st.warning("⚠️ **Incomplete Input:** You have pasted the **Job Description**, but the **Resume (PDF)** is missing.")

# --- LOGIC: STRICT MODE (TF-IDF ANALYSIS) ---
# Executes when both inputs are present AND 'Strict' mode is selected
if st.session_state.cv_text and job_description \
    and st.session_state.info and mode.lower() == "strict":

    try:
        # 1. INITIALIZE VECTORIZER
        # Use TF-IDF with English stop words to remove common filler words (e.g., "the", "and")
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # 2. FIT & TRANSFORM
        # Fit to JD to establish the vocabulary "Ground Truth"
        desc_vectors = vectorizer.fit_transform([job_description])
        # Transform CV based ONLY on the JD's vocabulary
        cv_vectors = vectorizer.transform([st.session_state.cv_text])

        # Display file processing info if available
        if st.session_state.info:
            st.info(st.session_state.info)

        # 3. CALCULATE MATCH SCORE
        # Compute Cosine Similarity between the CV vector and JD vector
        similarity_scores = cosine_similarity(cv_vectors, desc_vectors)[0][0]
        
        # Display the result
        st.metric("ATS Match Score (TF-IDF)", value = similarity_scores)

        # 4. PASS/FAIL THRESHOLD LOGIC
        if similarity_scores > 0.5:
            st.success("✅ **ATS Optimized:** High keyword matching detected.")
        else:
            st.error("⚠️ **Optimization Needed:** Low keyword match. Consider adding more terms from the JD.")

        # 5. IDENTIFY MISSING KEYWORDS
        # Extract all words (features) found in the Job Description
        desc_keywords = vectorizer.get_feature_names_out()
        # Get the frequency/score of these words in the CV
        cv_array = cv_vectors.toarray()[0]

        # Create a DataFrame to compare JD keywords vs CV content
        df = pd.DataFrame({
            "Keywords": desc_keywords,
            "score": cv_array
        })
        
        # Filter: Find keywords that exist in JD but have a score of 0 in the CV
        df_missing = df[df["score"] == 0]["Keywords"]

        # 6. DISPLAY RECOMMENDATIONS
        if not df_missing.empty:
            with st.expander("👀 View Missing Keywords"):
                # Display the list of missing words without the score column (cleaner UI)
                st.dataframe(
                    df_missing,   
                    use_container_width=True, 
                    hide_index=True
                )
        else:
            # Case where user has all keywords
            st.info("🎉 **Perfect Match:** No missing keywords detected based on the Job Description.")
            
    except Exception as e:
        error_msg = str(e).lower()
        answer = "" # Placeholder for the error message

        # 1. Handle Empty Vocabulary (Common TF-IDF Error)
        # Occurs if JD contains only filler words (e.g., "The and or") or is too short.
        if "empty vocabulary" in error_msg or "stop words" in error_msg:
            answer = "⚠️ **Insufficient Content (TF-IDF):**\n\nThe Job Description is too short or contains only common filler words (e.g., 'the', 'and'). The ATS scanner could not extract unique keywords.\n\n**Solution:** Please paste a longer, more detailed Job Description."

        # 2. Handle Data Dimension/Shape Errors
        # Rare, but can happen if text processing fails midway.
        elif "inconsistent" in error_msg or "dimension" in error_msg or "shape" in error_msg:
             answer = "📐 **Data Processing Error:**\n\nThere was a mismatch between the CV and Job Description data structures during calculation. Please try re-uploading the CV."

        # 3. Handle General/Unknown Errors
        else:
            answer = f"❌ **Analysis Error:**\n\nAn unexpected error occurred during the Strict Mode analysis.\n\n**Technical Details:** `{str(e)}`"

        # Display the specific error message to the user
        st.error(answer)

# --- LOGIC: FLEXIBLE MODE (AI + HYBRID ANALYSIS) ---
# Executes when inputs are present AND 'Flexibel' (Smart Mode) is selected
elif st.session_state.cv_text and job_description \
    and st.session_state.info and mode.lower() == "flexible":

    try:
        # 1. SEMANTIC EMBEDDING (AI ENGINE)
        # Convert text into vector embeddings to understand context/meaning
        desc_embeds = model.encode(job_description)
        cv_embeds = model.encode(st.session_state.cv_text)

        # Display file processing info if available
        if st.session_state.info:
            st.info(st.session_state.info)

        # 2. CALCULATE CONTEXTUAL SIMILARITY
        # Use Cosine Similarity on the AI embeddings
        # .item() converts the tensor result into a standard Python float
        similarity_scores = util.cos_sim(cv_embeds, desc_embeds).item()
        
        # Display the AI Score
        st.metric("AI Relevance Score (Context)", value = similarity_scores)

        # 3. INTERPRET SCORE
        if similarity_scores > 0.5:
            st.success("✅ **Strong Match:** The CV is contextually relevant to the job.")
        else:
            st.error("⚠️ **Low Relevance:** The CV content does not strongly align with the job context.")

        # 4. HYBRID ANALYSIS (TF-IDF FOR KEYWORDS)
        # Even in AI mode, we run TF-IDF purely to find specific missing keywords for the user.
        # This gives the user the "Best of Both Worlds" (Context Score + Keyword Advice).
        vectorizer = TfidfVectorizer(stop_words='english')
        vectorizer.fit([job_description])
        cv_vectors = vectorizer.transform([st.session_state.cv_text])

        # Extract features and scores
        desc_keywords = vectorizer.get_feature_names_out()
        cv_array = cv_vectors.toarray()[0]

        # Create DataFrame for analysis
        df = pd.DataFrame({
            "Keywords": desc_keywords,
            "score": cv_array
        })
        
        # Filter for words present in JD but missing in CV (score == 0)
        df_missing = df[df["score"] == 0]["Keywords"]

        # 5. DISPLAY MISSING KEYWORDS
        if not df_missing.empty:
            with st.expander("👀 View Missing Keywords (Hybrid Analysis)"):
                # Warning: Explains that while AI understands context, specific keywords help with strict ATS.
                st.warning("Tip: Although the AI understands your context, adding these specific keywords can help you pass strict ATS filters:")
                st.dataframe(
                    df_missing,   
                    use_container_width=True, 
                    hide_index=True
                )
        else:
            st.info("🎉 **Excellent Coverage:** All key terms from the job description are present.")
            
    except Exception as e:
        error_msg = str(e).lower()
        answer = "" # Placeholder for the error message

        # 1. Handle Empty Vocabulary (TF-IDF Error)
        # This occurs if the Job Description contains ONLY stop words (e.g., "The and or") and no unique keywords.
        if "empty vocabulary" in error_msg or "stop words" in error_msg:
            answer = "⚠️ **Insufficient Content (TF-IDF):**\n\nThe Job Description is too short or contains only common stop words (e.g., 'the', 'and'). The system could not extract valid keywords. Please paste a more detailed Job Description."

        # 2. Handle Memory/Resource Issues (AI Model Error)
        # This can happen if the text is massive or the server is overloaded.
        elif "cuda" in error_msg or "memory" in error_msg or "out of memory" in error_msg:
            answer = "💾 **System Limit Reached:**\n\nThe AI model encountered a memory limit while processing the text. Please try shortening the Job Description or refreshing the page."

        # 3. Handle Data/Shape Mismatch (Pandas/Vector Error)
        # Rare, but happens if vectors don't align during dataframe creation.
        elif "dimension" in error_msg or "shape" in error_msg or "length" in error_msg:
            answer = "📐 **Data Mismatch:**\n\nAn error occurred while matching the CV keywords with the Job Description. Please check if the inputs contain valid text."

        # 4. Handle General/Unknown Errors
        else:
            answer = f"❌ **Analysis Error:**\n\nAn unexpected error occurred during the Flexible Mode analysis.\n\n**Technical Details:** `{str(e)}`"

        # Display the specific error message to the user
        st.error(answer)