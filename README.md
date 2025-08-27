# RAG Chatbot

TOC-aware hybrid Retrieval-Augmented Generation (RAG) pipeline for PDFs using LangChain and Ollama.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Ask a single question:

```bash
python rag_pdf_ollama.py --pdf /path/manual.pdf --ask "What export formats are supported?"
```

Interactive session:

```bash
python rag_pdf_ollama.py --pdf /path/manual.pdf --interactive
```

You can specify alternative Ollama models via `--model` and `--embed`.

For direct module execution, ensure the `src` directory is on `PYTHONPATH` and run:

```bash
PYTHONPATH=src python -m rag_chatbot.cli --help
```
