import streamlit as st
import tempfile
import os
from RAG_pipeline_for_UI import process_pdfs, prepare_rag_index, query_rag_model

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
        faiss_index, all_chunks, _ = prepare_rag_index(all_chunks)

    st.success("✅ All documents processed successfully!")

    st.subheader("💬 Ask Questions About Your PDFs")
    user_query = st.text_input("Type your question:")

    if st.button("Submit Query") and user_query:
        with st.spinner("🤖 Thinking..."):
            answer = query_rag_model(user_query, faiss_index, all_chunks)
            st.success("🧠 Answer:")
            st.write(answer)
else:
    st.info("Please upload one or more PDF documents to begin.")
