"""Utilities for generating optional summaries of chunks and sections."""
from collections import defaultdict
from typing import Dict, List

from rag_chatbot.chunking import Chunk
from rag_chatbot.config import Config
from rag_chatbot.models import get_llm
from rag_chatbot.prompt_registry import registry


def build_summaries(chunks: List[Chunk], cfg: Config) -> Dict[str, Dict[str, str]]:
    """Generate summaries for individual chunks and whole sections.

    Returns a dictionary with two keys: ``chunks`` and ``sections`` mapping to
    summary strings.
    """
    llm = get_llm(cfg.llm_model, provider=cfg.llm_provider)

    chunk_summaries: Dict[str, str] = {}
    for ch in chunks:
        prompt = registry.get("summarize_chunk", "Summarize:\n{text}")
        chunk_summaries[ch.id] = llm.invoke(prompt.format(text=ch.text)).strip()

    by_section: Dict[str, List[Chunk]] = defaultdict(list)
    for ch in chunks:
        by_section[ch.heading_num].append(ch)

    section_summaries: Dict[str, str] = {}
    for num, chs in by_section.items():
        text = "\n\n".join(c.text for c in chs)
        title = chs[0].heading_title
        prompt = registry.get(
            "summarize_section", "Summarize section '{title}':\n{text}"
        )
        section_summaries[num] = llm.invoke(
            prompt.format(title=title, text=text)
        ).strip()

    return {"chunks": chunk_summaries, "sections": section_summaries}
