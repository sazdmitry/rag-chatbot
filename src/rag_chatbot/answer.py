from typing import List, Tuple

from langchain_ollama import OllamaLLM

from rag_chatbot.chunking import Chunk
from rag_chatbot.index import Index
from rag_chatbot.prompt_registry import registry
from rag_chatbot.retrieval import (
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
        if not ch or len(kept) >= ix.cfg.max_context_chunks:
            continue
        size = len(ch.text) + 200
        if total + size > ix.cfg.max_context_chars:
            continue
        kept.append(ch)
        total += size
    return kept


def render_context(keep: List[Chunk]) -> str:
    return "\n\n".join(
        f"### {ch.toc_path} {ch.citation()}\n{ch.text}" for ch in keep
    )


def answer_query(ix: Index, query: str) -> Tuple[str, List[Chunk]]:
    variants = multi_query_expand(query, ix.cfg)

    dense_rankings = [dense_search(ix, v, ix.cfg.topk_dense) for v in variants]
    bm25_rankings = [bm25_search(ix, v, ix.cfg.topk_bm25) for v in variants]

    fused = rrf_fuse(dense_rankings + bm25_rankings, k=ix.cfg.rrf_k)
    expanded = expand_neighborhood(ix, fused)
    reranked = maybe_rerank(ix, query, expanded)

    kept = pack_context(ix, reranked)
    ctx = render_context(kept)
    llm = OllamaLLM(model=ix.cfg.llm_model)
    sys_prompt = registry["answer_system"]
    user_prompt = registry["answer_user"].format(query=query, ctx=ctx)
    full_prompt = f"<|system|>\n{sys_prompt}\n<|user|>\n{user_prompt}"
    ans = llm.invoke(full_prompt)
    return ans.strip(), kept
