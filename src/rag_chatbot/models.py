from typing import Any


def get_llm(model_name: str, provider: str = "ollama", **kwargs: Any):
    """Return an LLM instance for the given provider."""
    if provider == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(model=model_name, **kwargs)
    elif provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(model_id=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_embeddings(model_name: str, provider: str = "ollama", **kwargs: Any):
    """Return an embeddings model for the given provider."""
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=model_name, **kwargs)
    elif provider == "bedrock":
        from langchain_aws import BedrockEmbeddings

        return BedrockEmbeddings(model_id=model_name, **kwargs)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def get_cross_encoder(model_name: str, **kwargs: Any):
    """Return a sentence-transformers CrossEncoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, **kwargs)
