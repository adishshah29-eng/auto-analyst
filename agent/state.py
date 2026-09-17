"""Structured, running state for one analysis pass.

Kept deliberately small: the prompt sent to the LLM on any given step is
built from this state, never from the raw dataset. See README "Context
Management" — a running structured summary in, not a growing transcript of
every intermediate value.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict


class PlannedStep(TypedDict):
    stage: str  # "clean" | "explore"
    description: str  # plain English, no code — what to do and why


class Finding(TypedDict):
    kind: str  # e.g. "distribution", "correlation", "outlier", "groupby"
    description: str
    stats: dict[str, Any]
    caveat: str  # non-empty when agent.agents.significance flags low n / weak effect size; "" otherwise


class CriticReview(TypedDict):
    kept: int
    dropped: int
    reasons: list[str]  # one short reason per dropped finding/chart, for the UI/log


class ChartMeta(TypedDict):
    path: str
    chart_type: str
    question: str  # what question this chart answers


class CodeStep(TypedDict):
    stage: str
    code: str
    success: bool
    error: str | None
    retried: bool


class AnalysisState(TypedDict):
    dataset_name: str
    run_id: str  # ties this run to its outputs/runs/<run_id>.jsonl trace file, see agent.tracing
    dataset_schema: dict[str, Any]
    suggested_questions: list[str]  # Planner's schema-aware starting points for the human to pick from
    user_goal: str  # what the human actually wants to know; steers explore, chart, and synthesis
    plan: list[PlannedStep]  # from the Planner agent; empty until planned
    plan_approved: bool  # False while awaiting human review (HITL gate)
    cleaning_actions_taken: list[str]
    findings: list[Finding]
    charts_generated: list[ChartMeta]
    critic_review: CriticReview | None
    narrative_review: dict[str, Any] | None  # LLM-as-judge score, see agent.agents.critic.review_narrative
    code_history: list[CodeStep]
    total_cost_usd: float
    total_tokens: dict[str, int]
    stage_timings_s: dict[str, float]
    narrative_summary: str


def new_state(dataset_name: str, run_id: str = "") -> AnalysisState:
    return AnalysisState(
        dataset_name=dataset_name,
        run_id=run_id,
        dataset_schema={},
        suggested_questions=[],
        user_goal="",
        plan=[],
        plan_approved=False,
        cleaning_actions_taken=[],
        findings=[],
        charts_generated=[],
        critic_review=None,
        narrative_review=None,
        code_history=[],
        total_cost_usd=0.0,
        total_tokens={"input": 0, "output": 0},
        stage_timings_s={},
        narrative_summary="",
    )


def record_code_step(
    state: AnalysisState, stage: str, code: str, success: bool, error: str | None, retried: bool
) -> None:
    # Trim: keep at most the last 20 steps in memory/prompts, older ones are
    # dropped rather than left to grow the context unbounded.
    state["code_history"].append(
        CodeStep(stage=stage, code=code, success=success, error=error, retried=retried)
    )
    if len(state["code_history"]) > 20:
        state["code_history"] = state["code_history"][-20:]


class StageTimer:
    """Context manager that records a stage's wall-clock duration into state."""

    def __init__(self, state: AnalysisState, stage: str):
        self.state = state
        self.stage = stage
        self._start = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.state["stage_timings_s"][self.stage] = time.monotonic() - self._start


def summarize_for_prompt(state: AnalysisState, max_findings: int = 15, max_charts: int = 10) -> dict[str, Any]:
    """The only view of `state` that ever goes into an LLM prompt.

    Deliberately excludes code_history bodies (kept for debugging/UI only)
    and any raw dataframe values.
    """
    return {
        "user_goal": state["user_goal"],
        "dataset_schema": state["dataset_schema"],
        "cleaning_actions_taken": state["cleaning_actions_taken"],
        "findings": state["findings"][-max_findings:],
        "charts_generated": [
            {"chart_type": c["chart_type"], "question": c["question"]}
            for c in state["charts_generated"][-max_charts:]
        ],
    }
