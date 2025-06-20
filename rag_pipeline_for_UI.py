import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from PIL import ImageEnhance
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

#RAG
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#poppler_path = r"C:path-to\Release-24.08.0-0\poppler-24.08.0\Library\bin"
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def pdf_to_images(pdf_path, dpi=300):
    #return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return convert_from_path(pdf_path, dpi=300)

def preprocess_image(img):
    img = img.convert("L")
    width, height = img.size
    img = img.resize((int(width * 1.5), int(height * 1.5)))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) 
    return img

def extract_text_with_tesseract(images):
    processed_text = []
    for img in images:
        enhanced_img = preprocess_image(img)
        text = pytesseract.image_to_string(enhanced_img)
        processed_text.append(text)
    return "\n\n".join(processed_text)

def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_chunks(chunks):
    return embedder.encode(chunks)

def check_ocr_quality(text, pdf_path, min_length=100):
    if not text or len(text.strip()) < min_length:
        raise ValueError(f"OCR extraction failed for document '{pdf_path}' due to low quality (text too short: {len(text)} characters).")
    return True

def build_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def ask_gemini_rag(question, retrieved_chunks):
    context = "\n---\n".join(retrieved_chunks)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are an intelligent assistant designed to analyze and answer questions from scanned documents.\n\n"
        "Your capabilities include:\n"
        "- Answering natural language questions using only the provided document chunks.\n"
        "- Understanding follow-up and context-aware questions (chat format).\n"
        "- Performing basic analytics (e.g., counting words, listing entities, finding totals).\n"
        "- Summarizing sections or comparing details across the document.\n"
        "- Responding 'The answer is not found in the documents.' if unsure or unsupported.\n\n"
        "Respond clearly, concisely, and helpfully using only the given document context.\n"
        f"Document Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Final Answer:"
    )
    response = model.generate_content(prompt)
    return response.text

#UI pipeline
def process_pdfs(pdf_paths):
    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            images = pdf_to_images(pdf_path)
            text = extract_text_with_tesseract(images)
            cleaned = clean_text(text)
            check_ocr_quality(cleaned, pdf_path) 
            chunks = chunk_text(cleaned)
            all_chunks.extend(chunks) 
                
        except ValueError as e:
            print(str(e))
            print("Extraction failed") 
        except Exception as e:
            print(f"Unexpected error processing {pdf_path}: {e}")
        
    return all_chunks

def prepare_rag_index(all_chunks):
    if not all_chunks:
        raise ValueError("No valid chunks were extracted. Cannot build FAISS index.")
    embeddings = embed_chunks(all_chunks)
    embeddings = np.array(embeddings)

    if embeddings.size == 0 or len(embeddings.shape) < 2:
        raise ValueError("Embedding failed or returned empty. Cannot build FAISS index.")
    index = build_faiss_index(embeddings)
    return index, all_chunks, embeddings

def query_rag_model(query, index, all_chunks):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    retrieved_chunks = [all_chunks[i] for i in I[0]]
    return ask_gemini_rag(query, retrieved_chunks)
