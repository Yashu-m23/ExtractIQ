# ExtractIQ: Document Intelligence & RAG-Powered Chat

## Overview

ExtractIQ is a Streamlit-based AI application that processes **image-based PDFs** using OCR, embeds their content, and enables intelligent question-answering via **RAG (Retrieval-Augmented Generation)** using Google's Gemini.

## 🚀 Live Demo

👉 [Check out the app here!](https://extractiq.streamlit.app/)

---

## Features

- Upload and process multiple scanned PDF files
- Optical Character Recognition (OCR) using Tesseract
- Enhanced OCR accuracy with preprocessing (grayscale, contrast, resizing)
- Chat using natural language
- Gemini 1.5 Flash for contextual understanding
- FAISS-powered vector search for relevant info
- Automatic detection of low-quality PDFs
- Caching to improve repeated query response speed
- UI with animated background and gradients

---

## How It Works

1. **Upload PDFs** → multiple scanned PDFs are supported
2. **OCR** → Extract text from each page using Tesseract
3. **Chunk & Embed** → Text is chunked and encoded into vectors
4. **Store in FAISS** → Fast vector search over chunks
5. **Chat with your documents** → Each message is matched to relevant chunks, and Gemini responds conversationally based on the context

---

## Tech Stack

- **Python 3.9+**
- Streamlit — UI
- Tesseract OCR
- Poppler — PDF rendering
- FAISS — vector similarity search
- Google Gemini — LLM-based Q&A
- SentenceTransformers — Embedding model

---

(File Structure
📁 App/
  ├── UI.py
  └── rag_pipeline_for_UI.py)
  
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yashu-m23/ExtractIQ.git
cd ExtractIQ
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run UI.py
```
