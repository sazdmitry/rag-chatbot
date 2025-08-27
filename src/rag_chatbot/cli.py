import argparse
import os
import sys

from .answer import answer_query
from .chunking import build_chunks
from .config import Config
from .index import build_index
from .pdf_utils import load_pdf_text


def main() -> None:
    parser = argparse.ArgumentParser(description="TOC‑aware Hybrid RAG for PDFs (LangChain + Ollama)")
    parser.add_argument("--pdf", help="Path to the PDF user manual", default="data/Manual_ver3.pdf")
    parser.add_argument("--ask", help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop")
    parser.add_argument("--model", help="Ollama LLM model (default llama3.2:3b)")
    parser.add_argument("--embed", help="Ollama embedding model (default bge-m3, fallback nomic-embed-text)")
    parser.add_argument("--toc-pages", type=int, default=0, help="Number of initial table-of-contents pages to parse")
    parser.add_argument("--footer-regex", help="Regular expression to remove footer text from each page")
    args = parser.parse_args()

    cfg = Config()
    if args.model:
        cfg.llm_model = args.model
    if args.embed:
        cfg.embed_model_primary = args.embed
    if args.toc_pages:
        cfg.toc_pages = args.toc_pages
    if args.footer_regex:
        cfg.footer_regex = args.footer_regex

    if not os.path.exists(args.pdf):
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    print("[1/4] Reading PDF …", file=sys.stderr)
    pages = load_pdf_text(args.pdf)

    print("[2/4] Building chunks …", file=sys.stderr)
    chunks = build_chunks(pages, cfg)
    print(f"  chunks: {len(chunks)}")

    print("[3/4] Building hybrid index (FAISS + BM25) …", file=sys.stderr)
    ix = build_index(chunks, cfg)

    def ask(q: str) -> None:
        print("\n[4/4] Retrieving & answering …", file=sys.stderr)
        ans, used = answer_query(ix, q)
        print("\n=== ANSWER ===\n")
        print(ans)
        print("\n--- Sources ---")
        seen = set()
        for ch in used:
            key = (ch.toc_path, ch.page_start, ch.page_end)
            if key in seen:
                continue
            seen.add(key)
            print(f"• {ch.citation()}")

    if args.ask:
        ask(args.ask)
    if args.interactive:
        print("\nInteractive mode. Type your question (or 'exit').\n")
        while True:
            try:
                q = input("? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            ask(q)


if __name__ == "__main__":
    main()
