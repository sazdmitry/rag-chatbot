from dataclasses import dataclass


@dataclass
class Config:
    """Runtime configuration for the RAG pipeline."""

    # Models
    llm_model: str = "llama3.2:3b"
    llm_provider: str = "ollama"
    embed_model: str = "bge-m3"
    embed_provider: str = "ollama"

    # PDF pre-processing
    toc_pages: int = 0  # number of initial table-of-contents pages
    footer_regex: str = ""

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
    reranker_type: str = "cross-encoder"  # "llm" or "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_provider: str = "hf"

    # Context packing
    max_context_chars: int = 9000
    max_context_chunks: int = 12

    # Prompting
    n_query_expansions: int = 4
