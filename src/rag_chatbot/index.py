import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from rag_chatbot.config import Config
from rag_chatbot.chunking import Chunk
from rag_chatbot.models import get_embeddings

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class Index:
    cfg: Config
    faiss: FAISS
    id_lookup: List[str]
    bm25: BM25Okapi
    bm25_corpus_tokens: List[List[str]]
    bm25_id_lookup: List[str]
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    siblings_by_parent: Dict[str, List[str]] = field(default_factory=dict)
    chunk_summaries: Dict[str, str] = field(default_factory=dict)
    section_summaries: Dict[str, str] = field(default_factory=dict)


def build_index(
    chunks: List[Chunk],
    cfg: Config,
    chunk_summaries: Optional[Dict[str, str]] = None,
    section_summaries: Optional[Dict[str, str]] = None,
) -> Index:
    chunk_summaries = chunk_summaries or {}
    section_summaries = section_summaries or {}

    texts = []
    metadatas = []
    for c in chunks:
        text_parts = [c.text]
        if c.id in chunk_summaries:
            text_parts.append(chunk_summaries[c.id])
        sec_sum = section_summaries.get(c.heading_num)
        if sec_sum:
            text_parts.append(sec_sum)
        texts.append("\n".join(text_parts))
        meta = {
            "id": c.id,
            "toc_path": c.toc_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "heading_num": c.heading_num,
            "heading_title": c.heading_title,
            "heading_level": c.heading_level,
            "ordinal_in_section": c.ordinal_in_section,
            "parent_key": c.parent_key,
        }
        if c.id in chunk_summaries:
            meta["chunk_summary"] = chunk_summaries[c.id]
        if sec_sum:
            meta["section_summary"] = sec_sum
        metadatas.append(meta)

    embeddings = get_embeddings(cfg.embed_model, provider=cfg.embed_provider)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    corpus_tokens = [TOKEN_PATTERN.findall(t.lower()) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    id_lookup = [m.id for m in vectorstore.docstore._dict.values()]  # type: ignore
    bm25_id_lookup = id_lookup[:]

    sibs: Dict[str, List[str]] = {}
    by_parent: Dict[str, List[Chunk]] = {}
    for c in chunks:
        by_parent.setdefault(c.parent_key, []).append(c)
    for pk, lst in by_parent.items():
        sibs[pk] = [c.id for c in lst]

    chunk_map = {c.id: c for c in chunks}

    return Index(
        cfg=cfg,
        faiss=vectorstore,
        id_lookup=id_lookup,
        bm25=bm25,
        bm25_corpus_tokens=corpus_tokens,
        bm25_id_lookup=bm25_id_lookup,
        chunks=chunk_map,
        siblings_by_parent=sibs,
        chunk_summaries=chunk_summaries,
        section_summaries=section_summaries,
    )


def save_index(ix: Index, path: str) -> None:
    """Persist an index to disk."""
    os.makedirs(path, exist_ok=True)
    ix.faiss.save_local(os.path.join(path, "faiss"))
    meta = {
        "cfg": asdict(ix.cfg),
        "id_lookup": ix.id_lookup,
        "bm25_corpus_tokens": ix.bm25_corpus_tokens,
        "bm25_id_lookup": ix.bm25_id_lookup,
        "chunks": [asdict(ch) for ch in ix.chunks.values()],
        "siblings_by_parent": ix.siblings_by_parent,
        "chunk_summaries": ix.chunk_summaries,
        "section_summaries": ix.section_summaries,
    }
    with open(os.path.join(path, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def load_index(path: str) -> Index:
    """Load a previously saved index from disk."""
    with open(os.path.join(path, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    cfg = Config(**meta["cfg"])
    embeddings = get_embeddings(cfg.embed_model, provider=cfg.embed_provider)
    faiss_store = FAISS.load_local(os.path.join(path, "faiss"), embeddings, allow_dangerous_deserialization=True)

    corpus_tokens = meta["bm25_corpus_tokens"]
    bm25 = BM25Okapi(corpus_tokens)
    chunk_objs = {c["id"]: Chunk(**c) for c in meta["chunks"]}

    return Index(
        cfg=cfg,
        faiss=faiss_store,
        id_lookup=meta["id_lookup"],
        bm25=bm25,
        bm25_corpus_tokens=corpus_tokens,
        bm25_id_lookup=meta["bm25_id_lookup"],
        chunks=chunk_objs,
        siblings_by_parent=meta["siblings_by_parent"],
        chunk_summaries=meta.get("chunk_summaries", {}),
        section_summaries=meta.get("section_summaries", {}),
    )
