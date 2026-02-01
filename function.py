import streamlit as st
from sentence_transformers import SentenceTransformer

def init_state():
    if "info" not in st.session_state:
        st.session_state.info = ""
    if "cv_text" not in st.session_state:
        st.session_state.cv_text = ""

def clear_cv():
    st.session_state.info = ""
    st.session_state.cv_text = ""

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')