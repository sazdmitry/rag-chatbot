import math
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag_chatbot.user_manual.config import Config


@dataclass
class Chunk:
    id: str
    text: str
    toc_path: str
    page_start: int
    page_end: int
    heading_num: str
    heading_title: str
    heading_level: int
    ordinal_in_section: int
    parent_key: str
    prev_id: Optional[str] = None
    next_id: Optional[str] = None

    def citation(self) -> str:
        pages = f"p{self.page_start}–{self.page_end}" if self.page_start != self.page_end else f"p{self.page_start}"
        return f"[{self.toc_path} — {pages}]"


TOC_ENTRY_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\s+(?P<title>.*?)\.\s+(?P<page>\d+)\s*$", re.MULTILINE)


def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def join_hyphenated_line_breaks(s: str) -> str:
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", s)


def token_len(s: str) -> int:
    # rough token proxy ~4 chars/token
    return max(1, math.ceil(len(s) / 4))


def split_paragraphs(s: str) -> List[str]:
    parts = re.split(r"\n{2,}", s)
    merged: List[str] = []
    buff: List[str] = []
    for p in parts:
        if re.match(r"^\s*([\-*•\d]+\.|[\-*•])\s+", p):
            buff.append(p)
            continue
        if buff:
            merged.append("\n\n".join(buff))
            buff = []
        merged.append(p)
    if buff:
        merged.append("\n\n".join(buff))
    cleaned: List[str] = []
    for x in merged:
        j = join_hyphenated_line_breaks(x)
        if normalize_ws(j):
            cleaned.append(normalize_ws(j.replace("\n", " ")))
    return cleaned


def parse_toc(pages: List[Tuple[int, str]], n_pages: int):
    """Extract heading entries from the table of contents pages."""
    text = "\n".join(t for _, t in pages[:n_pages])
    entries = []
    for m in TOC_ENTRY_RE.finditer(text):
        try:
            page = int(m.group("page"))
        except ValueError:
            continue
        entries.append({
            "num": m.group("num").strip(),
            "title": m.group("title").strip(),
            "page": page,
        })
    return entries


def build_toc_paths(headings: List[Dict[str, Any]]) -> Dict[str, str]:
    title_by_num: Dict[str, str] = {}
    for h in headings:
        title_by_num[h["num"]] = h["title"]

    path_by_num: Dict[str, str] = {}
    for num in title_by_num:
        parts = num.split(".")
        crumbs = []
        for i in range(1, len(parts) + 1):
            pnum = ".".join(parts[:i])
            title = title_by_num.get(pnum)
            if title:
                crumbs.append(f"{pnum} — {title}")
        path_by_num[num] = " › ".join([c.split(" — ", 1)[1] for c in crumbs]) if crumbs else num
    return path_by_num


def parent_key(num: str) -> str:
    parts = num.split(".")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def chunk_section(text: str, base_meta: Dict[str, Any], cfg: Config) -> List[Chunk]:
    paras = split_paragraphs(text)
    chunks: List[Chunk] = []
    buf: List[str] = []
    tokens = 0
    idx_in_sec = 0

    def flush():
        nonlocal buf, tokens, idx_in_sec, chunks
        if not buf:
            return
        chunk_text = "\n\n".join(buf).strip()
        if not chunk_text:
            buf = []
            tokens = 0
            return
        cid = str(uuid.uuid4())
        chunk = Chunk(
            id=cid,
            text=chunk_text,
            toc_path=base_meta["toc_path"],
            page_start=base_meta["page_start"],
            page_end=base_meta["page_end"],
            heading_num=base_meta["heading_num"],
            heading_title=base_meta["heading_title"],
            heading_level=base_meta["heading_level"],
            ordinal_in_section=idx_in_sec,
            parent_key=base_meta["parent_key"],
        )
        chunks.append(chunk)
        idx_in_sec += 1
        if cfg.atomic_chunk_overlap_tokens > 0 and len(buf) > 0:
            keep = []
            tks = 0
            for p in reversed(buf):
                tks += token_len(p)
                keep.append(p)
                if tks >= cfg.atomic_chunk_overlap_tokens:
                    break
            buf = list(reversed(keep))
            tokens = sum(token_len(x) for x in buf)
        else:
            buf = []
            tokens = 0

    for p in paras:
        p_tokens = token_len(p)
        if tokens + p_tokens > cfg.atomic_chunk_tokens and tokens > 0:
            flush()
        buf.append(p)
        tokens += p_tokens
    flush()
    return chunks


def build_chunks(pages: List[Tuple[int, str]], cfg: Config) -> List[Chunk]:
    """Build content chunks, optionally seeding headings from TOC pages."""
    if cfg.footer_regex:
        footer_re = re.compile(cfg.footer_regex, re.MULTILINE)
        pages = [(p, footer_re.sub("", t)) for p, t in pages]

    toc_headings: List[Dict[str, Any]] = []
    body_pages = pages
    if cfg.toc_pages > 0:
        toc_headings = parse_toc(pages, cfg.toc_pages)
        body_pages = pages[cfg.toc_pages:]

    page_map = {p: t for p, t in body_pages}

    headings: List[Dict[str, Any]] = []
    if toc_headings:
        for h in toc_headings:
            txt = page_map.get(h["page"], "")
            pat = re.compile(rf"^\s*{re.escape(h['num'])}\s+{re.escape(h['title'])}", re.MULTILINE)
            m = pat.search(txt)
            span = m.span() if m else (0, 0)
            h["span"] = span
            headings.append(h)

    if not headings:
        full_text = "\n\n".join([t for _, t in body_pages])
        base = {
            "toc_path": "Document",
            "page_start": body_pages[0][0] if body_pages else 1,
            "page_end": body_pages[-1][0] if body_pages else 1,
            "heading_num": "0",
            "heading_title": "Document",
            "heading_level": 0,
            "parent_key": "",
        }
        return chunk_section(full_text, base, cfg)

    headings.sort(key=lambda h: (h["page"], h["span"][0]))
    path_by_num = build_toc_paths(headings)

    sections: List[Tuple[Dict[str, Any], str, int, int]] = []

    for idx, h in enumerate(headings):
        start_page = h["page"]
        start_off = h["span"][1]
        end_page = body_pages[-1][0]
        end_off = len(page_map[end_page])
        if idx + 1 < len(headings):
            nh = headings[idx + 1]
            end_page = nh["page"]
            end_off = nh["span"][0]
        if start_page == end_page:
            body = page_map.get(start_page, "")[start_off:end_off]
        else:
            parts = [page_map.get(start_page, "")[start_off:]]
            for p in range(start_page + 1, end_page):
                parts.append(page_map.get(p, ""))
            parts.append(page_map.get(end_page, "")[:end_off])
            body = "\n".join(parts)
        body = body.strip()
        sections.append((h, body, start_page, end_page))

    all_chunks: List[Chunk] = []
    for h, body, pstart, pend in sections:
        meta = {
            "toc_path": path_by_num.get(h["num"], h["title"]),
            "page_start": pstart,
            "page_end": pend,
            "heading_num": h["num"],
            "heading_title": h["title"],
            "heading_level": len(h["num"].split(".")),
            "parent_key": parent_key(h["num"]),
        }
        chs = chunk_section(body, meta, cfg)
        all_chunks.extend(chs)

    for i, ch in enumerate(all_chunks):
        if i > 0:
            ch.prev_id = all_chunks[i - 1].id
        if i < len(all_chunks) - 1:
            ch.next_id = all_chunks[i + 1].id
    return all_chunks
