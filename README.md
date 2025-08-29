# RAG Chatbot

TOC-aware hybrid Retrieval-Augmented Generation (RAG) pipeline for PDFs using LangChain and Ollama.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess documents (run once)

```bash
python preprocess_pdf.py --pdf /path/manual.pdf --output data/index
```

Use `--config path.json` to override defaults from a JSON config file. Add
`--summary` to generate optional summaries for chunks and sections. The
preprocessing step builds the FAISS/BM25 index and stores it on disk so it can
be reused by the chat CLI.

### 2. Query the chatbot

Ask a single question:

```bash
python rag_pdf_ollama.py --index data/index --ask "What export formats are supported?"
```

Interactive session:

```bash
python rag_pdf_ollama.py --index data/index --interactive
```

The chat CLI allows overriding the answering model and provider via
`--model` and `--llm-provider`.

For direct module execution, ensure the `src` directory is on `PYTHONPATH` and run:

```bash
PYTHONPATH=src python -m rag_chatbot.cli --help
```
