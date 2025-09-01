import sys
from pathlib import Path

# Ensure src package is on path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatbot.user_manual.answer import answer_query
from rag_chatbot.user_manual.index import load_index


def print_answer(ix, question):
    """Answer a question and print citations once."""
    ans, chunks = answer_query(ix, question)
    print(ans)
    print("-- Sources --")
    seen = set()
    for ch in chunks:
        key = (ch.toc_path, ch.page_start, ch.page_end)
        if key in seen:
            continue
        seen.add(key)
        print(ch.citation())
    print()
    return ans, chunks
