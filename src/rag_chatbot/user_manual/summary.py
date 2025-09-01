"""Utilities for generating optional summaries of chunks and sections."""
from collections import defaultdict
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from rag_chatbot.common.prompt_registry import registry
from rag_chatbot.models import get_llm
from rag_chatbot.user_manual.chunking import Chunk
from rag_chatbot.user_manual.config import Config


def _heading_path(num: str, title_by_num: Dict[str, str]) -> str:
    parts = num.split(".")
    crumbs = []
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        title = title_by_num.get(prefix, "")
        crumbs.append(f"{prefix} {title}".strip())
    return " > ".join(crumbs)


def _page_span(start: int, end: int) -> str:
    return f"p. {start}" if start == end else f"pp. {start}–{end}"


class Summary(BaseModel):
    """Schema for summaries returned by the LLM."""

    heading_path: str
    overview: str
    key_actions: List[str] = Field(default_factory=list)
    ui_terms: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    synonyms: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    search_terms: List[str] = Field(default_factory=list)
    retrieval_text: str = ""


def build_summaries(chunks: List[Chunk], cfg: Config) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Generate structured summaries for individual chunks and whole sections."""

    llm = get_llm(cfg.llm_model, provider=cfg.llm_provider)
    structured = llm.with_structured_output(Summary)

    title_by_num: Dict[str, str] = {}
    for ch in chunks:
        title_by_num.setdefault(ch.heading_num, ch.heading_title)

    chunk_summaries: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        prompt = registry.get("summarize_chunk", "")
        hp = _heading_path(ch.heading_num, title_by_num)
        pages = _page_span(ch.page_start, ch.page_end)
        text = prompt.format(heading_path=hp, page_span=pages, text=ch.text)
        try:
            chunk_summaries[ch.id] = structured.invoke(text).dict()
        except Exception:
            resp = llm.invoke(text).content.strip()
            chunk_summaries[ch.id] = {"heading_path": hp, "overview": resp}

    by_section: Dict[str, List[Chunk]] = defaultdict(list)
    for ch in chunks:
        parts = ch.heading_num.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            by_section[prefix].append(ch)

    section_summaries: Dict[str, Dict[str, Any]] = {}
    for num, chs in by_section.items():
        text = "\n\n".join(c.text for c in chs)
        hp = _heading_path(num, title_by_num)
        pages = _page_span(min(c.page_start for c in chs), max(c.page_end for c in chs))
        prompt = registry.get("summarize_section", "")
        formatted = prompt.format(heading_path=hp, page_span=pages, text=text)
        try:
            section_summaries[num] = structured.invoke(formatted).dict()
        except Exception:
            resp = llm.invoke(formatted).content.strip()
            section_summaries[num] = {"heading_path": hp, "overview": resp}

    return {"chunks": chunk_summaries, "sections": section_summaries}
