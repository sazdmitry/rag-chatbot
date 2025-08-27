from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the RAG pipeline."""

    # Models
    llm_model: str = "llama3.2:3b"
    embed_model_primary: str = "bge-m3"
    embed_model_fallback: str = "nomic-embed-text"

    # Chunking
    atomic_chunk_tokens: int = 320
    atomic_chunk_overlap_tokens: int = 48
    max_section_tokens: int = 1100

    # Retrieval
    topk_dense: int = 8
    topk_bm25: int = 8
    rrf_k: int = 60
    expand_neighbors: int = 1
    expand_siblings: bool = True

    # Reranker
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Context packing
    max_context_chars: int = 9000
    max_context_chunks: int = 12

    # Prompting
    n_query_expansions: int = 4
