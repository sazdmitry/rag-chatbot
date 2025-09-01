import re
from typing import Dict, List

from rapidfuzz import fuzz

from rag_chatbot.common.prompt_registry import registry
from rag_chatbot.models import get_llm
from rag_chatbot.user_manual.config import Config
from rag_chatbot.user_manual.index import Index
from rag_chatbot.user_manual.reranking import CrossEncoderReranker, LLMReranker

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def multi_query_expand(query: str, cfg: Config) -> List[str]:
    llm = get_llm(cfg.llm_model, provider=cfg.llm_provider)
    prompt = registry["multi_query_expand"].format(
        n_query_expansions=cfg.n_query_expansions, query=query
    )
    out = llm.invoke(prompt).content
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in out.splitlines() if re.sub(r"[ \t]+", " ", x).strip()]
    uniq: List[str] = []
    for s in lines:
        if all(fuzz.token_sort_ratio(s, t) < 90 for t in uniq):
            uniq.append(s)
    if not uniq:
        return [query]
    return [query] + uniq[: cfg.n_query_expansions]


def rrf_fuse(rankings: List[List[str]], k: int) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def dense_search(ix: Index, query: str, topk: int) -> List[str]:
    docs = ix.faiss.similarity_search(query, k=topk)
    return [d.metadata.get("id") for d in docs]


def bm25_search(ix: Index, query: str, topk: int) -> List[str]:
    tokens = TOKEN_PATTERN.findall(query.lower())
    scores = ix.bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    return [ix.bm25_id_lookup[i] for i in ranked]


def expand_neighborhood(ix: Index, base_ids: List[str]) -> List[str]:
    cfg = ix.cfg
    selected = set(base_ids)
    if cfg.expand_neighbors > 0:
        pos = {cid: i for i, cid in enumerate(ix.id_lookup)}
        for cid in list(selected):
            if cid not in pos:
                continue
            i = pos[cid]
            for d in range(1, cfg.expand_neighbors + 1):
                if i - d >= 0:
                    selected.add(ix.id_lookup[i - d])
                if i + d < len(ix.id_lookup):
                    selected.add(ix.id_lookup[i + d])
    if cfg.expand_siblings:
        for cid in list(selected):
            ch = ix.chunks.get(cid)
            if not ch:
                continue
            sibs = ix.siblings_by_parent.get(ch.parent_key, [])
            for s in sibs:
                selected.add(s)
    return list(selected)


def maybe_rerank(ix: Index, query: str, candidate_ids: List[str]) -> List[str]:
    if not (ix.cfg.use_reranker and candidate_ids):
        return candidate_ids

    rtype = getattr(ix.cfg, "reranker_type", "cross-encoder")
    provider = getattr(ix.cfg, "reranker_provider", "hf")
    cache_folder = getattr(ix.cfg, "cache_folder", "data/cache")
    if rtype == "llm":
        reranker = LLMReranker(ix.cfg.reranker_model, provider=provider)
    else:
        reranker = CrossEncoderReranker(ix.cfg.reranker_model, cache_folder=cache_folder)

    return reranker.rerank(ix, query, candidate_ids)
