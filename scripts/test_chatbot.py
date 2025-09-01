import argparse

from chatbot_utils import load_index, print_answer

PREDEFINED_QUESTIONS = [
    "What export formats are supported?",
    "How do I reset the device?",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run chatbot on predefined questions"
    )
    parser.add_argument("--index", default="data/index", help="Path to index directory")
    args = parser.parse_args()

    ix = load_index(args.index)
    for q in PREDEFINED_QUESTIONS:
        print(f"=== Query: {q}")
        print_answer(ix, q)


if __name__ == "__main__":
    main()
