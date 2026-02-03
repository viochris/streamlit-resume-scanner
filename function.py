import streamlit as st
from collections import Counter
from sentence_transformers import SentenceTransformer
import nltk, re
from nltk.corpus import stopwords

def init_state():
    """
    Initializes necessary variables in Streamlit's session state.
    This ensures that 'info' and 'cv_text' keys exist before the app tries to access them,
    preventing KeyErrors during the first run.
    """
    # Initialize 'info' to store metadata messages (e.g., page count)
    if "info" not in st.session_state:
        st.session_state.info = ""
    
    # Initialize 'cv_text' to store the extracted text from the PDF
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""

def clear_cv():
    """
    Callback function to reset the stored CV data.
    This is triggered via the 'on_change' parameter in the file_uploader.
    It ensures that when a user uploads a new file, the old text data is wiped.
    """
    # Reset the info message
    st.session_state.info = ""
    # Clear the extracted CV text
    st.session_state.cv_text = ""

@st.cache_resource
def load_model():
    """
    Loads the SBERT (Sentence-BERT) model for semantic similarity analysis.
    
    @st.cache_resource:
    This decorator is crucial. It tells Streamlit to load this heavy model 
    ONLY ONCE and cache it in memory. Without this, the app would reload 
    the model (taking several seconds) every time the user clicks a button.

    It also performs a one-time check for NLTK stopwords. Placing this here 
    ensures that the necessary data is downloaded exactly once at startup, 
    preventing redundant checks during app usage.
    """
    # --- NLTK ONE-TIME SETUP ---
    # Check if 'stopwords' data is already available locally.
    # If not found (LookupError), download it immediately.
    # We place this inside @st.cache_resource so it runs only once per session,
    # preventing repetitive download checks on every interaction.
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Using 'paraphrase-multilingual-MiniLM-L12-v2' for good performance across languages
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')