import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

#RAG
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
poppler_path = r"C:\Users\yasha\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
genai.configure(api_key="AIzaSyBTuLzATYpmVjRHpUIUnb3aSpnESeqcbk4")

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

def extract_text_with_tesseract(images):
    return "\n\n".join(pytesseract.image_to_string(img) for img in images)

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

def build_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

def ask_gemini_rag(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"You are an assistant helping answer questions from scanned documents.\n\n"
        f"Relevant Document Chunks:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    response = model.generate_content(prompt)
    return response.text

#UI pipeline
def process_pdfs(pdf_paths):
    all_chunks = []
    for pdf_path in pdf_paths:
        images = pdf_to_images(pdf_path)
        text = extract_text_with_tesseract(images)
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)
    return all_chunks

def prepare_rag_index(all_chunks):
    embeddings = embed_chunks(all_chunks)
    embeddings = np.array(embeddings)
    index = build_faiss_index(embeddings)
    return index, all_chunks, embeddings

def query_rag_model(query, index, all_chunks):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    retrieved_chunks = [all_chunks[i] for i in I[0]]
    return ask_gemini_rag(query, retrieved_chunks)
