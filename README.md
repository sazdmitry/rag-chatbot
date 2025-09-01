# RAG Chatbot

TOC-aware hybrid Retrieval-Augmented Generation (RAG) pipeline for PDFs using LangChain and Ollama.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess documents (run once)

```bash
python scripts/preprocess_pdf.py --pdf /path/manual.pdf --output data/index
```

Use `--config path.json` to override defaults from a JSON config file. Add
`--summary` to generate structured JSON summaries for chunks and sections. The
summaries use LangChain's `with_structured_output` to enforce valid JSON. The
preprocessing step builds the FAISS/BM25 index and stores it on disk so it can
be reused by the chat CLI.

### 2. Query the chatbot

Start an interactive session:

```bash
python scripts/rag_pdf_ollama.py --index data/index
```

The chat CLI allows overriding the answering model and provider via
`--model` and `--llm-provider`.
