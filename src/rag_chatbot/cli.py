import argparse
import sys

from rag_chatbot.answer import answer_query
from rag_chatbot.index import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="TOC‑aware Hybrid RAG CLI")
    parser.add_argument("--index", default="data/index", help="Path to preprocessed index")
    parser.add_argument("--ask", help="Single question to ask")
    parser.add_argument("--interactive", action="store_true", help="Interactive Q&A loop")
    parser.add_argument("--model", help="LLM model override")
    parser.add_argument("--llm-provider", help="LLM provider override (ollama or bedrock)")
    args = parser.parse_args()

    print("[1/1] Loading index …", file=sys.stderr)
    ix = load_index(args.index)
    cfg = ix.cfg
    if args.model:
        cfg.llm_model = args.model
    if args.llm_provider:
        cfg.llm_provider = args.llm_provider

    def ask(q: str) -> None:
        print("\nRetrieving & answering …", file=sys.stderr)
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
