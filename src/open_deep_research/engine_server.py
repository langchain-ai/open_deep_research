"""FastAPI HTTP server exposing the deep_researcher graph as a generic engine.

Wraps the langgraph `deep_researcher` with a POST /research endpoint speaking
the neutral DeepResearchEngineRequest / DeepResearchEngineResult contract.
Bearer auth via OPEN_DEEP_RESEARCH_API_KEY; X-Request-ID echoed back as
runtime_trace_ref. Engine-side schema maps the langgraph final state into the
neutral result; consulting flavor is supplied later by deployment overlays
(see Raochra's RSM-308).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import HumanMessage

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.engine_models import (
    CandidateInsight,
    ConsultedArtifact,
    DeepResearchEngineRequest,
    DeepResearchEngineResult,
    EngineErrorResponse,
    ExecutedQueryNode,
)

logger = logging.getLogger(__name__)

_API_KEY_ENV = "OPEN_DEEP_RESEARCH_API_KEY"


app = FastAPI(
    title="Open Deep Research engine",
    description=(
        "Generic deep-research engine exposing the neutral "
        "DeepResearchEngineRequest / DeepResearchEngineResult contract."
    ),
)


def _check_bearer(authorization: str | None) -> None:
    """Validate the Bearer header against the configured API key."""
    expected = os.environ.get(_API_KEY_ENV, "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail=f"server is not configured: {_API_KEY_ENV} is unset",
        )
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="missing Authorization header",
        )
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="malformed Authorization header (expected 'Bearer <key>')",
        )
    if parts[1] != expected:
        raise HTTPException(
            status_code=401,
            detail="invalid API key",
        )


async def authenticate(
    authorization: str | None = Header(default=None),
) -> None:
    _check_bearer(authorization)


@app.exception_handler(HTTPException)
async def _typed_error(_: Request, exc: HTTPException) -> JSONResponse:
    """Map HTTPException codes to the neutral error envelope."""
    if exc.status_code in (401, 403):
        error_type = "authority_limited"
    elif exc.status_code == 429:
        error_type = "budget_exhausted"
    elif exc.status_code in (400, 422):
        error_type = "schema_invalid"
    else:
        error_type = "internal_error"
    body = EngineErrorResponse(error_type=error_type, detail=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/research",
    response_model=DeepResearchEngineResult,
    dependencies=[Depends(authenticate)],
)
async def research(
    payload: DeepResearchEngineRequest,
    request: Request,
    x_request_id: str | None = Header(default=None),
) -> JSONResponse:
    runtime_trace_ref = x_request_id or str(uuid.uuid4())
    started_at = time.monotonic()
    logger.info(
        "research request",
        extra={
            "task_id": payload.task_id,
            "trigger_ref": payload.trigger_ref,
            "runtime_trace_ref": runtime_trace_ref,
            "target_kind": payload.research_target.target_kind,
        },
    )

    try:
        graph_input = {
            "messages": [HumanMessage(content=payload.research_target.question)]
        }
        graph_result = await deep_researcher.ainvoke(graph_input)
    except Exception as exc:
        logger.exception("graph execution failed")
        raise HTTPException(
            status_code=500,
            detail=f"deep_researcher graph failure: {exc}",
        ) from exc

    elapsed_seconds = max(int(time.monotonic() - started_at), 0)
    final_report: str = (
        graph_result.get("final_report") or graph_result.get("notes", [""])[-1] or ""
    ).strip()

    seed = payload.research_plan_seed[0]
    executed = ExecutedQueryNode(
        question=payload.research_target.question,
        research_mode=seed.research_mode,
        source_domain=seed.allowed_source_domains[0],
        retrieval_strategy=seed.allowed_retrieval_strategies[0],
        generated_sub_questions=[],
        budget_spent_sources=0,
        budget_spent_sub_queries=0,
        budget_spent_tokens=0,
        elapsed_seconds=elapsed_seconds,
        stop_reason="completed",
    )
    consulted = ConsultedArtifact(
        source_reference="langgraph:deep_researcher",
        source_type="synthetic",
        source_domain=seed.allowed_source_domains[0],
        retrieval_timestamp=datetime.now(tz=UTC),
        source_credibility="inferred",
        applied_access_label=payload.access_context.access_label,
        produced_evidence=bool(final_report),
        notes=(
            "Engine produced a synthesized report; structured source "
            "extraction is overlay-side work (RSM-308)."
        ),
    )
    findings: list[CandidateInsight] = []
    if final_report:
        findings.append(
            CandidateInsight(
                statement=final_report[:1000],
                source_lineage_refs=["langgraph:deep_researcher"],
                derivation_basis="cross_source_synthesis",
                source_credibility="inferred",
                applied_access_label=payload.access_context.access_label,
                insight_kind="synthesized",
            )
        )

    result = DeepResearchEngineResult(
        task_id=payload.task_id,
        trigger_ref=payload.trigger_ref,
        runtime_trace_ref=runtime_trace_ref,
        executed_query_tree=[executed],
        consulted_sources=[consulted],
        candidate_findings=findings,
    )

    headers = {"X-Request-ID": runtime_trace_ref}
    return JSONResponse(content=result.model_dump(mode="json"), headers=headers)
