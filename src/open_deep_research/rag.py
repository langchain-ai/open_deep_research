"""Local RAG retrieval tools for personalized research workflows."""

import asyncio
import os
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from open_deep_research.configuration import Configuration, EmbeddingProvider


def _resolve_openai_api_key(config: RunnableConfig | None) -> str | None:
    """Resolve OpenAI API key from environment or runtime configuration."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true" and config:
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("OPENAI_API_KEY")
    return os.getenv("OPENAI_API_KEY")


def _resolve_openai_base_url(config: RunnableConfig | None) -> str | None:
    """Resolve OpenAI-compatible base URL from runtime configuration or environment."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true" and config:
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        base_url = api_keys.get("OPENAI_BASE_URL") or api_keys.get("OPENAI_API_BASE")
        if base_url:
            return base_url
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def _normalize_embedding_model(model_name: str) -> str:
    """Normalize an embedding model string for provider-specific SDKs."""
    if model_name.startswith("openai:"):
        return model_name.split(":", 1)[1]
    return model_name


def _resolve_embedding_provider(configurable: Configuration) -> EmbeddingProvider:
    """Resolve provider, allowing model-name based auto-detection for compatibility."""
    normalized_model = _normalize_embedding_model(configurable.embedding_model).lower()
    if (
        configurable.embedding_provider == EmbeddingProvider.OPENAI
        and normalized_model.startswith("nvidia/")
    ):
        return EmbeddingProvider.NVIDIA
    return configurable.embedding_provider


class _NvidiaEmbeddingsWrapper:
    """Embedding wrapper that injects NVIDIA-specific extra_body by operation type."""

    def __init__(self, base_embeddings: Any, configurable: Configuration) -> None:
        self._base_embeddings = base_embeddings
        self._configurable = configurable

    def _extra_body(self, input_type: str) -> dict[str, str]:
        return {
            "input_type": input_type,
            "truncate": self._configurable.embedding_truncate,
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "extra_body": self._extra_body(self._configurable.embedding_input_type_document)
        }
        if self._configurable.embedding_encoding_format.strip():
            kwargs["encoding_format"] = self._configurable.embedding_encoding_format
        return self._base_embeddings.embed_documents(texts, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        kwargs: dict[str, Any] = {
            "extra_body": self._extra_body(self._configurable.embedding_input_type_query)
        }
        if self._configurable.embedding_encoding_format.strip():
            kwargs["encoding_format"] = self._configurable.embedding_encoding_format
        return self._base_embeddings.embed_query(text, **kwargs)


def _build_embeddings(configurable: Configuration, config: RunnableConfig | None):
    """Build embedding function for the configured provider."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for local RAG embeddings. Install with `pip install langchain-openai`."
        ) from exc

    normalized_model = _normalize_embedding_model(configurable.embedding_model)
    provider = _resolve_embedding_provider(configurable)
    base_url = _resolve_openai_base_url(config)

    common_kwargs: dict[str, Any] = {
        "model": normalized_model,
        "api_key": _resolve_openai_api_key(config),
    }
    if base_url:
        common_kwargs["base_url"] = base_url

    if provider == EmbeddingProvider.NVIDIA:
        # NVIDIA compatible endpoints expect raw string inputs plus provider-specific request fields.
        base_embeddings = OpenAIEmbeddings(
            **common_kwargs,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
        return _NvidiaEmbeddingsWrapper(base_embeddings, configurable)

    return OpenAIEmbeddings(**common_kwargs)


def _build_vectorstore(configurable: Configuration, config: RunnableConfig | None):
    """Construct a Chroma vector store client from runtime configuration."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise ImportError(
            "langchain-chroma is required for local RAG. Install with `pip install langchain-chroma`."
        ) from exc

    embeddings = _build_embeddings(configurable, config)

    persist_dir = Path(configurable.chroma_persist_directory)
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="open_deep_research_local_kb",
    )


def _format_documents_for_output(query: str, docs: list) -> str:
    """Format retrieved documents into a stable tool output string."""
    if not docs:
        return f"Query: {query}\nNo local knowledge base results found."

    lines: list[str] = [f"Query: {query}"]
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_path", "unknown")
        updated_at = doc.metadata.get("updated_at", "unknown")
        content = str(doc.page_content).strip()
        lines.append(f"\n--- LOCAL SOURCE {i} ---")
        lines.append(f"Source: {source}")
        lines.append(f"UpdatedAt: {updated_at}")
        lines.append(f"Content:\n{content}")

    return "\n".join(lines)


@tool(description="Search the local knowledge base for personalized context and prior documents.")
async def rag_search(queries: list[str], config: RunnableConfig | None = None) -> str:
    """Run semantic search over local Chroma knowledge base for one or more queries."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.rag_enabled:
        return "Local RAG is disabled. Set rag_enabled=true to use this tool."

    if not queries or not any(str(query).strip() for query in queries):
        return "No valid queries were provided to rag_search."

    persist_dir = Path(configurable.chroma_persist_directory)
    if not persist_dir.exists():
        return (
            "Local vector store does not exist yet. Build it first with "
            "`python -m open_deep_research.ingestion --source <folder> --rebuild`."
        )

    if not persist_dir.is_dir():
        return "Local vector store path exists but is not a directory."

    if not any(persist_dir.iterdir()):
        return (
            "Local vector store directory is empty. Run ingestion first with "
            "`python -m open_deep_research.ingestion --source <folder> --rebuild`."
        )

    vectorstore = _build_vectorstore(configurable, config)

    async def _search_single_query(query: str) -> str:
        docs = await asyncio.to_thread(
            vectorstore.similarity_search,
            query,
            configurable.rag_top_k,
        )
        return _format_documents_for_output(query, docs)

    results = await asyncio.gather(*[_search_single_query(query) for query in queries])
    return "\n\n".join(results)
