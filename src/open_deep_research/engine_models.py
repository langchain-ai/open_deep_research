"""Engine-facing request/result schemas for the /research HTTP boundary.

These models mirror the neutral deep-research engine contract that
sensemaker-studio's ConsultingResearcher adapter compiles requests against
and parses results from. Field names and types must stay wire-compatible
with sensemaker_studio.consulting_workbench.intelligence.research_engine.

Engine vocabulary only — no consulting concepts. Domain shaping
(consulting framework constraints, source-authority tiers, knowledge
layers) flows in via request fields and is treated generically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


EngineExecutionMode = Literal["auto", "approve_each", "off"]
EngineSourceDomain = Literal[
    "local_context",
    "external_source",
    "reference_material",
    "pattern_library",
]
EngineSourceCredibility = Literal[
    "primary",
    "practitioner",
    "secondary",
    "inferred",
]
EngineTargetKind = Literal[
    "gap",
    "hypothesis",
    "contradiction",
    "missing_referent",
    "temporal_ambiguity",
    "source_authentication",
    "disconfirming_check",
]
EngineDerivationBasis = Literal[
    "direct_extraction",
    "cross_source_synthesis",
    "interpretive_inference",
]
EngineInsightKind = Literal["stated", "synthesized", "interpretive"]
EngineEscalationReason = Literal[
    "underspecified_task",
    "authority_limited",
    "budget_exhausted",
    "conflicting_sources",
    "insufficient_source_credibility",
    "attachment_required",
    "connector_gap",
    "doctrine_pressure",
]
ResearchMode = Literal[
    "signal_interpretation",
    "framework_lookup",
    "fact_extraction",
    "pattern_recognition",
    "synthesis",
]
ResearchRetrievalStrategy = Literal[
    "lexical",
    "semantic",
    "hybrid",
    "disconfirming",
    "graph_walk",
]
ResearchStopReason = Literal[
    "completed",
    "budget_exhausted",
    "redundant_results",
    "authority_limited",
    "schema_invalid",
    "operator_pause",
]
SourceType = Literal["public", "internal", "operator_attached", "synthetic"]


class ResearchBudget(BaseModel):
    """Limits the engine honors per task."""

    max_sources: int = Field(ge=0)
    max_sub_queries: int = Field(ge=0)
    max_depth: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    max_elapsed_seconds: int = Field(ge=0)


class TopicLensScope(BaseModel):
    """Topic-lens envelope the engine may honor during planning and synthesis."""

    scope_label: str | None = None
    allowed_lenses: list[str] = Field(default_factory=list)


class EngineAccessContext(BaseModel):
    """Access envelope applied before external material enters engine results."""

    actor_role: str = Field(min_length=1)
    access_label: str = Field(min_length=1)
    access_ceiling: str | None = None
    audience_label: str | None = None


class ResearchTarget(BaseModel):
    """Exact research target the engine is allowed to pursue."""

    target_kind: EngineTargetKind
    question: str = Field(min_length=1)
    supporting_context: str | None = None
    disconfirming_required: bool = False


class QueryPlanSeedNode(BaseModel):
    """One allowed branch in the query-plan seed."""

    question: str = Field(min_length=1)
    research_mode: ResearchMode
    allowed_source_domains: list[EngineSourceDomain] = Field(min_length=1)
    allowed_retrieval_strategies: list[ResearchRetrievalStrategy] = Field(min_length=1)


class ExecutedQueryNode(BaseModel):
    """One executed node in the engine query tree."""

    question: str = Field(min_length=1)
    research_mode: ResearchMode
    source_domain: EngineSourceDomain
    retrieval_strategy: ResearchRetrievalStrategy
    generated_sub_questions: list[str] = Field(default_factory=list)
    budget_spent_sources: int = Field(ge=0)
    budget_spent_sub_queries: int = Field(ge=0)
    budget_spent_tokens: int = Field(ge=0)
    elapsed_seconds: int = Field(ge=0)
    stop_reason: ResearchStopReason


class ConsultedArtifact(BaseModel):
    """Record of a consulted source, including no-finding rows."""

    source_reference: str = Field(min_length=1)
    source_type: SourceType
    source_domain: EngineSourceDomain
    retrieval_timestamp: datetime
    source_credibility: EngineSourceCredibility
    applied_access_label: str = Field(min_length=1)
    produced_evidence: bool
    notes: str = ""


class CandidateInsight(BaseModel):
    """A proposed finding emitted by one engine task."""

    statement: str = Field(min_length=1)
    source_lineage_refs: list[str] = Field(min_length=1)
    derivation_basis: EngineDerivationBasis
    source_credibility: EngineSourceCredibility
    applied_access_label: str = Field(min_length=1)
    insight_kind: EngineInsightKind


class DisconfirmingInsight(BaseModel):
    """Material that weakens or contradicts the active line of inquiry."""

    statement: str = Field(min_length=1)
    source_lineage_refs: list[str] = Field(min_length=1)
    notes: str = ""


class StructuredGap(BaseModel):
    """A structured abstention emitted by the engine."""

    what_examined: str
    what_sought: str
    why_not_found: str
    what_would_resolve: str


class SourceArtifactAttachment(BaseModel):
    """Externally retrieved material proposed for explicit attachment."""

    source_reference: str = Field(min_length=1)
    source_type: SourceType
    source_domain: EngineSourceDomain
    applied_access_label: str = Field(min_length=1)
    attachment_reason: str = Field(min_length=1)


class EngineEscalation(BaseModel):
    """An operator-facing escalation emitted by the engine."""

    reason: EngineEscalationReason
    detail: str = Field(min_length=1)
    smallest_next_action: str = Field(min_length=1)


class DeepResearchEngineRequest(BaseModel):
    """Engine-facing scoped research request (POST /research body)."""

    task_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    operator_id: str = Field(min_length=1)
    execution_mode: EngineExecutionMode
    trigger_ref: str = Field(min_length=1)
    research_target: ResearchTarget
    research_plan_seed: list[QueryPlanSeedNode] = Field(min_length=1)
    budget: ResearchBudget
    access_context: EngineAccessContext
    source_preferences: dict[str, object] = Field(default_factory=dict)
    task_context: dict[str, object] = Field(default_factory=dict)
    preference_context: dict[str, object] = Field(default_factory=dict)
    topic_lens_scope: TopicLensScope = Field(default_factory=TopicLensScope)


class DeepResearchEngineResult(BaseModel):
    """Engine-facing raw research result (POST /research response body)."""

    task_id: str = Field(min_length=1)
    trigger_ref: str = Field(min_length=1)
    runtime_trace_ref: Optional[str] = None
    executed_query_tree: list[ExecutedQueryNode] = Field(min_length=1)
    consulted_sources: list[ConsultedArtifact] = Field(min_length=1)
    candidate_findings: list[CandidateInsight] = Field(default_factory=list)
    disconfirming_material: list[DisconfirmingInsight] = Field(default_factory=list)
    surfaced_tensions: list[str] = Field(default_factory=list)
    unknowns: list[StructuredGap] = Field(default_factory=list)
    candidate_source_artifact_attachments: list[SourceArtifactAttachment] = Field(
        default_factory=list
    )
    escalations: list[EngineEscalation] = Field(default_factory=list)


class EngineErrorResponse(BaseModel):
    """Typed error envelope emitted on /research failures."""

    error_type: Literal[
        "authority_limited",
        "budget_exhausted",
        "schema_invalid",
        "internal_error",
    ]
    detail: str = Field(min_length=1)
