# ExtractIQ: Document Intelligence & RAG-Powered Chat

ExtractIQ is a Streamlit-based AI application that processes **image-based PDFs** using OCR, embeds their content, and enables intelligent question-answering via **RAG (Retrieval-Augmented Generation)** using Google's Gemini.

---

## Features

- Multi-PDF upload support
- OCR via Tesseract for scanned documents
- Semantic chunking & embedding using Sentence Transformers
- FAISS vector search for fast retrieval
- Context-aware querying with Gemini (Gemini 1.5 Flash)
- Streamlit chat UI for document Q&A

---

## How It Works

1. **Upload PDFs** → multiple scanned PDFs are supported
2. **OCR** → Extract text from each page using Tesseract
3. **Chunk & Embed** → Text is chunked and encoded into vectors
4. **Store in FAISS** → Fast vector search over chunks
5. **Ask Questions** → Query is matched to chunks, and Gemini generates answers

---

## Tech Stack

- Streamlit
- Tesseract OCR
- Sentence-Transformers
- FAISS
- Gemini (Google Generative AI)

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
