import argparse
import sys

from chatbot_utils import load_index, print_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="TOC-aware Hybrid RAG CLI")
    parser.add_argument("--index", default="data/index", help="Path to preprocessed index")
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

    print("\nInteractive mode. Type your question (or 'exit').\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\nRetrieving & answering …", file=sys.stderr)
        print("\n=== ANSWER ===\n")
        print_answer(ix, q)


if __name__ == "__main__":
    main()
