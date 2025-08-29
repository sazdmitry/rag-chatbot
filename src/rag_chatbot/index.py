import re
from dataclasses import dataclass, field
from typing import Dict, List

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


def build_index(chunks: List[Chunk], cfg: Config) -> Index:
    texts = [c.text for c in chunks]
    metadatas = [{
        "id": c.id,
        "toc_path": c.toc_path,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "heading_num": c.heading_num,
        "heading_title": c.heading_title,
        "heading_level": c.heading_level,
        "ordinal_in_section": c.ordinal_in_section,
        "parent_key": c.parent_key,
    } for c in chunks]

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

    return Index(cfg=cfg, faiss=vectorstore, id_lookup=id_lookup, bm25=bm25,
                 bm25_corpus_tokens=corpus_tokens, bm25_id_lookup=bm25_id_lookup,
                 chunks=chunk_map, siblings_by_parent=sibs)
