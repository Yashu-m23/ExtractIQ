# ExtractIQ: Document Intelligence & RAG-Powered Chat

ExtractIQ is a Streamlit-based AI application that processes **image-based PDFs** using OCR, embeds their content, and enables intelligent question-answering via **RAG (Retrieval-Augmented Generation)** using Google's Gemini.

---

## Features

- Upload and process multiple scanned PDF files
- Optical Character Recognition (OCR) using Tesseract
- Enhanced OCR accuracy with preprocessing (grayscale, contrast, resizing)
- Ask questions using natural language
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
5. **Ask Questions** → Query is matched to chunks, and Gemini generates answers

---

## Tech Stack

- **Python 3.9+**
- [Streamlit](https://streamlit.io/) — UI
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler](https://poppler.freedesktop.org/) — PDF rendering
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search
- [Google Gemini](https://ai.google.dev/) — LLM-based Q&A
- [SentenceTransformers](https://www.sbert.net/) — Embedding model

---

(File Structure
📁 App/
  ├── UI.py
  └── RAG_pipeline_for_UI.py)
  
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
