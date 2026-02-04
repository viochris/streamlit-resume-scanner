# 📄 Resume Scanner Pro | ATS & AI Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![SBERT](https://img.shields.io/badge/SBERT-Sentence%20Transformers-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Overview
**Resume Scanner Pro** is a powerful **Dual-Engine CV Analysis tool** designed to help job seekers optimize their resumes for both automated **Applicant Tracking Systems (ATS)** and human recruiters.

Unlike simple keyword counters, this tool utilizes a hybrid approach:
1.  **🤖 Strict Mode (ATS Logic):** Uses statistical **TF-IDF (Term Frequency-Inverse Document Frequency)** to rigorously check for exact keyword matches, simulating older, strict ATS algorithms.
2.  **🧠 Flexible Mode (AI Logic):** Uses **Semantic Embeddings (SBERT)** to understand the *context* and *meaning* of skills. It recognizes that "Coding" and "Programming" are related, even if the words differ.

## ✨ Key Features

### 🚀 Dual-Engine Analysis
* **Strict Mode:** Perfect for "keyword-heavy" job applications. It calculates a match score based on exact vocabulary overlap using Scikit-Learn's TF-IDF.
* **Flexible Mode:** Perfect for modern applications. It uses a pre-trained AI model (`paraphrase-multilingual-MiniLM-L12-v2`) to measure how well the *meaning* of your CV matches the Job Description.

### 🔍 Hybrid Keyword Suggestion
Even when using the **AI Mode** to get a context score, the system runs a background **TF-IDF analysis** to provide actionable insights:
* **Missing Keywords:** Automatically identifies specific terms found in the Job Description but absent from your CV.
* **Smart Filtering:** Uses **NLTK Stopwords** to remove common filler words (e.g., *the, and, to*) and focus on high-value professional terms.

### 🎯 Critical Skills Check (Auto-Priority)
* **Top 5 Priority:** The system automatically identifies the most frequent and unique keywords in the Job Description.
* **Risk Alert:** It cross-references these top priorities with your CV. If you miss a critical hard skill (e.g., "SQL" for a Data role), the system issues a **"High Risk of Rejection"** warning.

### 🛡️ Robust Error Handling & Performance
* **Secure Processing:** Handles encrypted/password-protected PDFs gracefully with clear user alerts.
* **Resource Efficiency:** Uses `@st.cache_resource` to load the heavy AI model only once, ensuring the app runs fast after the first launch.
* **Data Validation:** Automatically detects empty files, corrupt headers, or insufficient text content.

### 📄 Intelligent PDF Parsing
* **Multi-Page Detection:** Detects if a resume exceeds 1 page and warns that recruiters may prioritize the first page.
* **Single-Page Optimization:** Confirms optimal length for quick recruiter scanning.

## 🛠️ Tech Stack
* **Frontend:** Streamlit.
* **NLP & AI:**
    * `Sentence-Transformers` (SBERT) for Semantic Embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).
    * `Scikit-Learn` (TF-IDF Vectorizer) for Statistical Analysis.
    * `NLTK` (Natural Language Toolkit) for Stopwords processing.
* **Data Processing:** Pandas, PyPDF2.
* **Math:** Cosine Similarity (Linear Kernel).

## ⚠️ Requirements & Limitations

### 1. File Format
* **PDF Only:** The system strictly accepts `.pdf` files to ensure accurate text extraction similar to real ATS environments.
* **Readable Text:** The PDF must contain selectable text (not scanned images/photos).

### 2. Language Support
* The system uses the `paraphrase-multilingual-MiniLM-L12-v2` model, which supports **50+ languages**, making it effective for English, Indonesian, and many other languages.
* *Note:* The "Missing Keywords" feature uses English stop-words filtering by default.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/viochris/streamlit-resume-scanner.git
    cd streamlit-resume-scanner
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🚀 Usage Guide

1.  **Upload Resume:**
    * Click the **"📂 Upload Resume (PDF)"** button on the left panel.
    * *Tip: Ensure your file is not password-protected.*

2.  **Input Job Description:**
    * Paste the full text of the job posting into the **"📋 Job Description"** text area on the right.

3.  **Select Mode:**
    * **Strict:** Choose this if you want to know exactly which words are missing.
    * **Flexible:** Choose this to see how well your experience aligns conceptually with the job.

4.  **Analyze:**
    * Click **"🔍 Analyze Match"**.
    * **Score:** View your match percentage (0-100%).
    * **Missing Keywords:** Expand the dropdown to see specific terms found in the Job Description but missing from your CV.
    * **Critical Skills:** Check the "Critical Skills Check" section to ensure you aren't missing mandatory requirements.

## 📷 Gallery

### 1. Landing Interface
![Home UI](assets/home_ui.png)
*The clean, balanced dual-column layout allows users to easily input their CV and the target Job Description side-by-side.*

### 2. Strict Mode Analysis (ATS Logic)
![Strict Mode](assets/used_ui_strict.png)
*The system calculates a strict similarity score based on exact vocabulary overlap, simulating rigid ATS filters.*

### 3. Flexible Mode Analysis (AI Logic)
![Flexible Mode](assets/used_ui_flexible.png)
*The AI Engine calculates a semantic relevance score. Even if keywords differ, the system understands the professional context.*

### 4. Actionable Insights & Critical Skills
![Missing Keywords](assets/missing_keywords.png)
*Regardless of the score, the system lists "Missing Keywords" and highlights "Top Priority" skills that are absent, offering a clear path to optimization.*

### 5. Critical Skills & Risk Alert
![Critical Skills](assets/critical_skills.png)
*The "Critical Skills Check" auto-detects the Top 5 mandatory requirements. If you miss these high-priority terms (e.g., specific hard skills), the system issues a "High Risk" warning to prevent immediate rejection.*

---
**Author:** [Silvio Christian, Joe](https://www.linkedin.com/in/silvio-christian-joe)
*"Don't let the ATS reject your dream job. Optimize it."*
