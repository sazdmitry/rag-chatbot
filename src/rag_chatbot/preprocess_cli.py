import argparse
import json
import os
import sys

from rag_chatbot.chunking import build_chunks
from rag_chatbot.config import Config
from rag_chatbot.index import build_index, save_index
from rag_chatbot.pdf_utils import load_pdf_text
from rag_chatbot.summary import build_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a PDF into a searchable index")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument(
        "--output", default="data/index", help="Directory to store the index"
    )
    parser.add_argument(
        "--config", help="Path to JSON config file overriding defaults"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Generate chunk and section summaries"
    )
    args = parser.parse_args()

    cfg = Config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if not os.path.exists(args.pdf):
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    print("[1/4] Reading PDF …", file=sys.stderr)
    pages = load_pdf_text(args.pdf)

    print("[2/4] Building chunks …", file=sys.stderr)
    chunks = build_chunks(pages, cfg)
    print(f"  chunks: {len(chunks)}", file=sys.stderr)

    chunk_sums = {}
    section_sums = {}
    if args.summary:
        print("[3/4] Generating summaries …", file=sys.stderr)
        sums = build_summaries(chunks, cfg)
        chunk_sums = sums["chunks"]
        section_sums = sums["sections"]

    print("[4/4] Building index …", file=sys.stderr)
    ix = build_index(chunks, cfg, chunk_sums, section_sums)

    save_index(ix, args.output)
    print(f"Index saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
