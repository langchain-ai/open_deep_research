from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from open_deep_research.configuration import Configuration

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(title: str) -> None:
    logger.info(f"\n=== {title} ===")


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_module(module_name: str) -> None:
    assert_true(
        importlib.util.find_spec(module_name) is not None,
        f"Missing required module: {module_name}",
    )


def validate_preflight() -> bool:
    print_header("Preflight")

    required_modules = [
        "langchain_chroma",
        "langchain_openai",
        "chromadb",
        "langchain_text_splitters",
    ]
    for module_name in required_modules:
        require_module(module_name)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    assert_true(
        bool(api_key),
        "OPENAI_API_KEY is required for embedding-based ingestion and retrieval acceptance.",
    )

    from open_deep_research.rag import _build_embeddings, _normalize_embedding_model

    embedding_model_env = os.getenv("EMBEDDING_MODEL", "").strip()
    embedding_model = _normalize_embedding_model(
        embedding_model_env or "openai:text-embedding-3-small"
    )
    embedding_provider_env = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if embedding_provider_env:
        embedding_provider = embedding_provider_env
    elif embedding_model.lower().startswith("nvidia/"):
        embedding_provider = "nvidia"
    else:
        embedding_provider = "openai"

    embedding_available = True
    try:
        cfg = Configuration.from_runnable_config(
            {
                "configurable": {
                    "embedding_model": embedding_model_env or "openai:text-embedding-3-small",
                    "embedding_provider": embedding_provider,
                    "embedding_encoding_format": os.getenv("EMBEDDING_ENCODING_FORMAT", "float"),
                    "embedding_input_type_query": os.getenv("EMBEDDING_INPUT_TYPE_QUERY", "query"),
                    "embedding_input_type_document": os.getenv("EMBEDDING_INPUT_TYPE_DOCUMENT", "passage"),
                    "embedding_truncate": os.getenv("EMBEDDING_TRUNCATE", "NONE"),
                }
            }
        )
        emb = _build_embeddings(cfg, None)
        emb.embed_query("acceptance_preflight")
    except Exception as exc:
        embedding_available = False
        logger.warning(
            f"Embedding endpoint unavailable for provider '{embedding_provider}' model '{embedding_model}'. "
            f"PR2 ingestion and PR3 RAG output tests will be skipped. "
            f"Error: {exc}"
        )

    logger.info("PASS: required modules and OPENAI_API_KEY are available")
    return embedding_available


def validate_pr1_config() -> None:
    print_header("PR1 Configuration Validation")

    ok = Configuration(rag_enabled=True, chroma_persist_directory=".chroma")
    assert_true(ok.rag_enabled is True, "Expected rag_enabled=True to be accepted")

    try:
        Configuration(rag_enabled=True, chroma_persist_directory="   ")
    except Exception as exc:
        msg = str(exc)
        assert_true(
            "chroma_persist_directory must be provided" in msg,
            "Expected explicit error for empty chroma_persist_directory",
        )
        logger.info("PASS: empty chroma_persist_directory is rejected")
    else:
        raise AssertionError("Expected validation error for empty chroma_persist_directory")

    try:
        Configuration(memory_enabled=True, memory_namespace_prefix="   ")
    except Exception as exc:
        msg = str(exc)
        assert_true(
            "memory_namespace_prefix must be provided" in msg,
            "Expected explicit error for empty memory_namespace_prefix",
        )
        logger.info("PASS: empty memory_namespace_prefix is rejected")
    else:
        raise AssertionError("Expected validation error for empty memory_namespace_prefix")



def _collection_count(persist_dir: Path) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection("open_deep_research_local_kb")
    return collection.count()


def validate_pr2_ingestion_cli(workspace: Path) -> Path:
    print_header("PR2 Ingestion CLI Behavior")

    fixture_root = workspace / "tests" / "pr1_pr3_acceptance" / "fixtures"
    runs_root = fixture_root / "chroma_runs"
    empty_dir = fixture_root / "empty_kb"
    text_dir = fixture_root / "text_kb"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    persist_dir = runs_root / f"run_{run_id}"

    empty_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    sample_file = text_dir / f"fact_{run_id}.txt"
    sample_file.write_text("DeepShark_001 from local kb", encoding="utf-8")

    env = os.environ.copy()
    env["CHROMA_PERSIST_DIRECTORY"] = str(persist_dir)
    env["RAG_ENABLED"] = "true"

    cmd_empty = [
        sys.executable,
        "-m",
        "open_deep_research.ingestion",
        "--source",
        str(empty_dir),
    ]
    res_empty = subprocess.run(
        cmd_empty,
        cwd=workspace,
        check=False,
        text=True,
        env=env,
        capture_output=True,
    )
    assert_true(
        res_empty.returncode != 0,
        "Expected non-zero exit when ingesting empty folder",
    )
    logger.info("PASS: empty folder ingestion returns non-zero exit")

    cmd_build = [
        sys.executable,
        "-m",
        "open_deep_research.ingestion",
        "--source",
        str(text_dir),
    ]
    res_build = subprocess.run(
        cmd_build,
        cwd=workspace,
        check=False,
        text=True,
        env=env,
        capture_output=True,
    )
    assert_true(
        res_build.returncode == 0,
        "Expected first ingestion run to succeed. "
        f"stdout={res_build.stdout[-600:]} stderr={res_build.stderr[-600:]}",
    )

    count_after_first = _collection_count(persist_dir)
    assert_true(count_after_first > 0, "Expected first ingestion run to write chunks")

    cmd_build_again = [
        sys.executable,
        "-m",
        "open_deep_research.ingestion",
        "--source",
        str(text_dir),
    ]
    res_build_again = subprocess.run(
        cmd_build_again,
        cwd=workspace,
        check=False,
        text=True,
        env=env,
        capture_output=True,
    )
    assert_true(
        res_build_again.returncode == 0,
        "Expected second ingestion run to succeed. "
        f"stdout={res_build_again.stdout[-600:]} stderr={res_build_again.stderr[-600:]}",
    )

    count_after_second = _collection_count(persist_dir)
    assert_true(
        count_after_second == count_after_first,
        (
            "Expected idempotent ingestion count; "
            f"first={count_after_first}, second={count_after_second}"
        ),
    )

    logger.info("PASS: ingestion succeeds and remains idempotent across repeated runs")
    return persist_dir



def validate_pr3_rag_tool_registration() -> None:
    print_header("PR3 RAG Tool Wiring")

    from open_deep_research.utils import get_all_tools

    config = {
        "configurable": {
            "rag_enabled": True,
            "search_api": "none",
            "chroma_persist_directory": ".chroma",
        }
    }

    import asyncio

    tools = asyncio.run(get_all_tools(config))
    tool_names = [t.name if hasattr(t, "name") else t.get("name") for t in tools]

    assert_true("rag_search" in tool_names, "Expected rag_search in assembled tools")

    if "think_tool" in tool_names:
        rag_idx = tool_names.index("rag_search")
        think_idx = tool_names.index("think_tool")
        assert_true(rag_idx > think_idx, "Expected rag_search to be registered after core tools")

    logger.info("PASS: rag_search is present in runtime tool list")


def validate_pr3_rag_output(persist_dir: Path) -> None:
    print_header("PR3 RAG Output Shape")

    import asyncio

    from open_deep_research.rag import rag_search

    config = {
        "configurable": {
            "rag_enabled": True,
            "chroma_persist_directory": str(persist_dir),
            "rag_top_k": 3,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "openai:text-embedding-3-small"),
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "openai"),
            "embedding_encoding_format": os.getenv("EMBEDDING_ENCODING_FORMAT", "float"),
            "embedding_input_type_query": os.getenv("EMBEDDING_INPUT_TYPE_QUERY", "query"),
            "embedding_input_type_document": os.getenv("EMBEDDING_INPUT_TYPE_DOCUMENT", "passage"),
            "embedding_truncate": os.getenv("EMBEDDING_TRUNCATE", "NONE"),
        }
    }

    previous_rag_enabled = os.environ.get("RAG_ENABLED")
    previous_chroma_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    os.environ["RAG_ENABLED"] = "true"
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(persist_dir)
    try:
        output = asyncio.run(
            rag_search.ainvoke(
                {"queries": ["DeepShark_001"]},
                config=config,
            )
        )
    finally:
        if previous_rag_enabled is None:
            os.environ.pop("RAG_ENABLED", None)
        else:
            os.environ["RAG_ENABLED"] = previous_rag_enabled

        if previous_chroma_dir is None:
            os.environ.pop("CHROMA_PERSIST_DIRECTORY", None)
        else:
            os.environ["CHROMA_PERSIST_DIRECTORY"] = previous_chroma_dir
    text = str(output)
    assert_true("Source:" in text, "Expected Source field in rag_search output")
    assert_true("UpdatedAt:" in text, "Expected UpdatedAt field in rag_search output")
    assert_true("DeepShark_001" in text, "Expected retrieved content to include fixture keyword")

    logger.info("PASS: rag_search output includes source metadata and retrieved content")



def main() -> int:
    workspace = Path(__file__).resolve().parents[2]

    dotenv_spec = importlib.util.find_spec("dotenv")
    if dotenv_spec is not None:
        from dotenv import load_dotenv

        load_dotenv(workspace / ".env")

    embedding_available = validate_preflight()
    validate_pr1_config()
    
    if embedding_available:
        persist_dir = validate_pr2_ingestion_cli(workspace)
        validate_pr3_rag_tool_registration()
        validate_pr3_rag_output(persist_dir)
    else:
        logger.info("SKIP: PR2 ingestion and PR3 RAG output tests (embedding unavailable)")
        validate_pr3_rag_tool_registration()

    logger.info("\nAll acceptance checks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
