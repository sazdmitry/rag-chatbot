from typing import List, Tuple

from langchain_community.llms import Ollama

from .chunking import Chunk
from .index import Index
from .retrieval import (
    bm25_search,
    dense_search,
    expand_neighborhood,
    maybe_rerank,
    multi_query_expand,
    rrf_fuse,
)


def pack_context(ix: Index, ordered_ids: List[str]) -> List[Chunk]:
    kept: List[Chunk] = []
    total = 0
    for cid in ordered_ids:
        ch = ix.chunks.get(cid)
        if not ch:
            continue
        if len(kept) >= ix.cfg.max_context_chunks:
            break
        if total + len(ch.text) + 200 > ix.cfg.max_context_chars:
            continue
        kept.append(ch)
        total += len(ch.text) + 200
    return kept


def render_context(keep: List[Chunk]) -> str:
    blocks = []
    for ch in keep:
        header = f"{ch.toc_path} {ch.citation()}"
        body = ch.text
        blocks.append(f"### {header}\n{body}")
    return "\n\n".join(blocks)


def answer_query(ix: Index, query: str) -> Tuple[str, List[Chunk]]:
    variants = multi_query_expand(query, ix.cfg)

    dense_rankings: List[List[str]] = []
    bm25_rankings: List[List[str]] = []
    for v in variants:
        dense_rankings.append(dense_search(ix, v, ix.cfg.topk_dense))
        bm25_rankings.append(bm25_search(ix, v, ix.cfg.topk_bm25))

    fused = rrf_fuse(dense_rankings + bm25_rankings, k=ix.cfg.rrf_k)
    expanded = expand_neighborhood(ix, fused)
    reranked = maybe_rerank(ix, query, expanded)

    kept = pack_context(ix, reranked)
    ctx = render_context(kept)

    llm = Ollama(model=ix.cfg.llm_model)
    sys_prompt = (
        "You answer questions using the provided manual excerpts. "
        "Cite after each claim with the chunk header in square brackets as already provided (e.g., [Export › CSV — p14]). "
        "If information is missing, say so."
    )
    user_prompt = f"""
Question: {query}

Use ONLY the following context. Summarize across sibling sections when needed, and enumerate options clearly.

{ctx}
""".strip()
    full_prompt = f"<|system|>\n{sys_prompt}\n<|user|>\n{user_prompt}"
    ans = llm.invoke(full_prompt)
    return ans.strip(), kept
