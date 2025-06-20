import streamlit as st

st.markdown(
    """
    <style>
    /* Fullscreen animated aurora gradient background */
    .stApp {
        background: linear-gradient(-45deg, #1b1b2f, #162447, #1f4068, #1b98e0, #23a6d5, #23d5ab);
        background-size: 600% 600%;
        animation: gradientFlow 20s ease infinite;
        color: white;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 100%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 0%; }
        100% { background-position: 0% 50%; }
    }

    /* Optional: Improve readability by styling containers */
    .block-container {
        background-color: rgba(0, 0, 0, 0.6); 
        padding: 2rem;
        border-radius: 12px;
    }

    h1, h2, h3, h4, h5, h6, label {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import tempfile
import os
from rag_pipeline_for_UI import process_pdfs, prepare_rag_index, query_rag_model
from streamlit import cache_data

import platform
import pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    poppler_path = r"C:\path-to\poppler\Library\bin"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    poppler_path = None

st.set_page_config(page_title="ExtractIQ", layout="wide")
st.title("📄 ExtractIQ: Multi-PDF Upload & Chat")

st.subheader("📂 Upload Your PDF Files")
uploaded_files = st.file_uploader(
    "Drag and drop multiple PDF files here or click to browse",
    type=["pdf"],
    accept_multiple_files=True
)

all_chunks = []
faiss_index = None

if uploaded_files:
    temp_paths = []

    with st.spinner("📥 Saving and processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_paths.append(tmp_file.name)

        all_chunks = process_pdfs(temp_paths)

    if not all_chunks:
        st.warning("⚠️ OCR extraction failed for all uploaded files. Please try uploading clearer PDFs.")
   
    else:
        faiss_index, all_chunks, _ = prepare_rag_index(all_chunks)
        st.success("✅ All documents processed successfully!")
        st.subheader("💬 Ask Questions About Your PDFs")
        user_query = st.text_input("Type your question:")

        @st.cache_data(show_spinner="🔁 Retrieving cached answer...")
        def cached_query(user_query, _faiss_index, _all_chunks):
            return query_rag_model(user_query, _faiss_index, _all_chunks)
        
        if st.button("Submit Query") and user_query:
            with st.spinner("🤖 Thinking..."):
                answer = cached_query(user_query, faiss_index, all_chunks)
                st.success("🧠 Answer:")
                st.write(answer)

else:
    st.info("Please upload one or more PDF documents to begin.")
