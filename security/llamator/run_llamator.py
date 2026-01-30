#!/usr/bin/env python3
"""
Run LLAMATOR security tests against OpenAI-compatible endpoints.

Expected environment:
- LIVE_API=1 (required to run live calls)
- OPENAI_API_KEY or LLM_API_KEY (API key)
- OPENAI_BASE_URL (optional; default https://api.openai.com/v1)

This script is intentionally self-contained and does not depend on project code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


# ---- LLAMATOR imports (new API first, then fallbacks) ----
def _import_llamator() -> tuple[Any, Any, Any, Any]:
    """
    Returns: (ClientOpenAI, ClientLangChain, start_testing, get_test_preset)

    We try the documented public API first. If the package layout differs,
    we fall back to plausible module paths.
    """
    # Documented usage shows importing from "llamator". :contentReference[oaicite:4]{index=4}
    try:
        import llamator  # type: ignore

        client_openai = getattr(llamator, "ClientOpenAI")
        client_lc = getattr(llamator, "ClientLangChain")
        start_testing = getattr(llamator, "start_testing")
        get_test_preset = getattr(llamator, "get_test_preset")
        return client_openai, client_lc, start_testing, get_test_preset
    except Exception:
        pass

    # Fallbacks for older/internal layouts (best-effort).
    try:
        from llamator.client import ClientOpenAI, ClientLangChain  # type: ignore
        from llamator import start_testing, get_test_preset  # type: ignore

        return ClientOpenAI, ClientLangChain, start_testing, get_test_preset
    except Exception as e:
        raise RuntimeError(
            "Не могу импортировать LLAMATOR API (ClientOpenAI/ClientLangChain/start_testing/get_test_preset). "
            "Проверь установленный llamator и его версию в текущем venv."
        ) from e


ClientOpenAI, ClientLangChain, start_testing, get_test_preset = _import_llamator()


@dataclass(frozen=True)
class ClientSpec:
    kind: str  # "openai" | "langchain"
    model: str
    base_url: str
    api_key: str
    temperature: float = 0.1
    # For langchain backend name, e.g. "ChatOpenAI"
    backend: str = "ChatOpenAI"
    extra: Optional[Dict[str, Any]] = None


def _require_live_api() -> None:
    if os.getenv("LIVE_API") != "1":
        raise SystemExit(
            "LIVE_API!=1. Этот запуск делает реальные запросы к API. "
            "Поставь LIVE_API=1 и проверь ключи в .env"
        )


def _load_env() -> None:
    if load_dotenv is not None:
        # .env in repo root is typical; allow override by DOTENV_PATH
        dotenv_path = os.getenv("DOTENV_PATH")
        if dotenv_path:
            load_dotenv(dotenv_path, override=False)
        else:
            load_dotenv(override=False)


def _get_api_key() -> str:
    # Prefer OPENAI_API_KEY, but allow LLM_API_KEY too
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
    if not api_key:
        raise SystemExit("Не найден API ключ. Укажи OPENAI_API_KEY (или LLM_API_KEY) в окружении/.env.")
    return api_key


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"


def _make_artifacts_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("results") / "llamator" / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_client(spec: ClientSpec) -> Any:
    """
    Builds a LLAMATOR client.

    ClientOpenAI signature in docs requires api_key, base_url, model. :contentReference[oaicite:5]{index=5}
    ClientLangChain signature requires backend name and kwargs. :contentReference[oaicite:6]{index=6}
    """
    if spec.kind == "openai":
        return ClientOpenAI(
            api_key=spec.api_key,
            base_url=spec.base_url,
            model=spec.model,
            temperature=spec.temperature,
        )

    if spec.kind == "langchain":
        kwargs: Dict[str, Any] = dict(spec.extra or {})
        # Common kwargs for ChatOpenAI-like backends
        kwargs.setdefault("model", spec.model)
        kwargs.setdefault("openai_api_key", spec.api_key)
        # Some backends accept base_url/openai_api_base; keep both best-effort
        kwargs.setdefault("base_url", spec.base_url)
        kwargs.setdefault("openai_api_base", spec.base_url)
        kwargs.setdefault("temperature", spec.temperature)

        return ClientLangChain(
            backend=spec.backend,
            **kwargs,
        )

    raise ValueError(f"Unknown client kind: {spec.kind}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLAMATOR live tests.")
    p.add_argument("--tested-model", default=os.getenv("LLAMATOR_TESTED_MODEL", "gpt-4o-mini"))
    p.add_argument("--attack-model", default=os.getenv("LLAMATOR_ATTACK_MODEL", "gpt-4o-mini"))
    p.add_argument("--judge-model", default=os.getenv("LLAMATOR_JUDGE_MODEL", "gpt-4o-mini"))

    p.add_argument("--tested-kind", choices=["openai", "langchain"], default=os.getenv("LLAMATOR_TESTED_KIND", "openai"))
    p.add_argument("--attack-kind", choices=["openai", "langchain"], default=os.getenv("LLAMATOR_ATTACK_KIND", "openai"))
    p.add_argument("--judge-kind", choices=["openai", "langchain"], default=os.getenv("LLAMATOR_JUDGE_KIND", "openai"))

    p.add_argument("--langchain-backend", default=os.getenv("LLAMATOR_LANGCHAIN_BACKEND", "ChatOpenAI"))

    p.add_argument("--base-url", default=_get_base_url())
    p.add_argument("--temperature", type=float, default=float(os.getenv("LLAMATOR_TEMPERATURE", "0.1")))

    p.add_argument(
        "--preset",
        default=os.getenv("LLAMATOR_TEST_PRESET", "owasp:llm01"),
        help="LLAMATOR test preset name (e.g. all, rus, eng, owasp:llm01, ...)",
    )
    p.add_argument("--threads", type=int, default=int(os.getenv("LLAMATOR_THREADS", "1")))
    p.add_argument("--report-language", choices=["en", "ru"], default=os.getenv("LLAMATOR_REPORT_LANG", "ru"))
    p.add_argument("--debug-level", type=int, choices=[0, 1, 2], default=int(os.getenv("LLAMATOR_DEBUG", "1")))
    p.add_argument("--enable-reports", action="store_true", default=(os.getenv("LLAMATOR_ENABLE_REPORTS", "1") == "1"))
    p.add_argument("--enable-logging", action="store_true", default=(os.getenv("LLAMATOR_ENABLE_LOGGING", "1") == "1"))
    return p.parse_args()


def main() -> int:
    _load_env()
    _require_live_api()

    api_key = _get_api_key()
    args = parse_args()

    artifacts_dir = _make_artifacts_dir()

    # Build clients
    tested = create_client(
        ClientSpec(
            kind=args.tested_kind,
            model=args.tested_model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            backend=args.langchain_backend,
        )
    )
    attack = create_client(
        ClientSpec(
            kind=args.attack_kind,
            model=args.attack_model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            backend=args.langchain_backend,
        )
    )
    judge = create_client(
        ClientSpec(
            kind=args.judge_kind,
            model=args.judge_model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            backend=args.langchain_backend,
        )
    )

    # Config is REQUIRED by start_testing() in current docs. :contentReference[oaicite:7]{index=7}
    config: Dict[str, Any] = {
        "enable_logging": bool(args.enable_logging),
        "enable_reports": bool(args.enable_reports),
        "artifacts_path": str(artifacts_dir),
        "debug_level": int(args.debug_level),
        "report_language": str(args.report_language),
    }

    # Basic tests preset (safe default).
    try:
        basic_tests: List[Tuple[str, Dict[str, Any]]] = get_test_preset(args.preset)
    except Exception as e:
        raise SystemExit(f"Не удалось загрузить preset '{args.preset}': {e}") from e

    results = start_testing(
        attack_model=attack,
        tested_model=tested,
        judge_model=judge,
        config=config,
        num_threads=args.threads,
        basic_tests=basic_tests,
        custom_tests=None,
    )

    out_path = artifacts_dir / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[llamator] done, artifacts: {artifacts_dir}")
    print(f"[llamator] results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
