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
        prompt = registry.get(
            "summarize_chunk",
            "Summarize the following text in one or two sentences. Respond with the summary only:\n{text}",
        )
        chunk_summaries[ch.id] = llm.invoke(prompt.format(text=ch.text)).strip()

    by_section: Dict[str, List[Chunk]] = defaultdict(list)
    title_by_num: Dict[str, str] = {}
    for ch in chunks:
        # record title for this heading number
        title_by_num.setdefault(ch.heading_num, ch.heading_title)
        parts = ch.heading_num.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            by_section[prefix].append(ch)

    section_summaries: Dict[str, str] = {}
    for num, chs in by_section.items():
        text = "\n\n".join(c.text for c in chs)
        title = title_by_num.get(num, chs[0].heading_title)
        prompt = registry.get(
            "summarize_section",
            "Provide a brief summary for the section titled '{title}'. Respond with the summary only:\n{text}",
        )
        section_summaries[num] = llm.invoke(
            prompt.format(title=title, text=text)
        ).strip()

    return {"chunks": chunk_summaries, "sections": section_summaries}
