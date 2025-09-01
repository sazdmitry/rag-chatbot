import argparse

from rag_chatbot.user_manual.answer import answer_query
from rag_chatbot.user_manual.index import load_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrieval pipeline against a prepared index"
    )
    parser.add_argument("index", help="Path to index directory")
    parser.add_argument("queries", nargs="+", help="Queries to test")
    args = parser.parse_args()

    ix = load_index(args.index)
    for q in args.queries:
        print(f"=== Query: {q}")
        ans, chunks = answer_query(ix, q)
        print(ans)
        print("-- Sources --")
        for ch in chunks:
            print(f"{ch.citation()}")
        print()


if __name__ == "__main__":
    main()
