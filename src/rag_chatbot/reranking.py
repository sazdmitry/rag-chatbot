import re
from typing import List

from rag_chatbot.index import Index
from rag_chatbot.models import get_llm, get_cross_encoder


class BaseReranker:
    """Base reranker interface."""

    def rerank(self, ix: Index, query: str, candidate_ids: List[str]) -> List[str]:
        raise NotImplementedError


class LLMReranker(BaseReranker):
    """Rerank using an LLM that scores relevance."""

    def __init__(self, model_name: str, provider: str = "ollama") -> None:
        self.llm = get_llm(model_name, provider=provider)

    def rerank(self, ix: Index, query: str, candidate_ids: List[str]) -> List[str]:
        pairs = [(cid, ix.chunks[cid].text) for cid in candidate_ids if cid in ix.chunks]
        if not pairs:
            return candidate_ids
        scored = []
        for cid, text in pairs:
            prompt = (
                "Given the query and document below, return a number between 0 and 1\n"
                f"Query: {query}\nDocument: {text}\nScore:"
            )
            try:
                out = self.llm.invoke(prompt).content
                match = re.search(r"[0-1]?\.\d+", out)
                score = float(match.group()) if match else 0.0
            except Exception:
                score = 0.0
            scored.append((score, cid))
        ranked = [cid for _, cid in sorted(scored, key=lambda x: x[0], reverse=True)]
        return ranked


class CrossEncoderReranker(BaseReranker):
    """Rerank using a sentence-transformers CrossEncoder model."""

    def __init__(self, model_name: str, **kwargs) -> None:
        self.model = get_cross_encoder(model_name, **kwargs)

    def rerank(self, ix: Index, query: str, candidate_ids: List[str]) -> List[str]:
        filtered_ids = [cid for cid in candidate_ids if cid in ix.chunks]
        if not filtered_ids:
            return candidate_ids
        pairs = [(query, ix.chunks[cid].text) for cid in filtered_ids]
        scores = self.model.predict(pairs)
        ranked = [cid for _, cid in sorted(zip(scores, filtered_ids), key=lambda x: x[0], reverse=True)]
        return ranked
