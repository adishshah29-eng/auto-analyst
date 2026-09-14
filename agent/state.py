"""Structured, running state for one analysis pass.

Kept deliberately small: the prompt sent to the LLM on any given step is
built from this state, never from the raw dataset. See README "Context
Management" — a running structured summary in, not a growing transcript of
every intermediate value.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict


class Finding(TypedDict):
    kind: str  # e.g. "distribution", "correlation", "outlier", "groupby"
    description: str
    stats: dict[str, Any]


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
    dataset_schema: dict[str, Any]
    cleaning_actions_taken: list[str]
    findings: list[Finding]
    charts_generated: list[ChartMeta]
    code_history: list[CodeStep]
    total_cost_usd: float
    total_tokens: dict[str, int]
    stage_timings_s: dict[str, float]
    narrative_summary: str


def new_state(dataset_name: str) -> AnalysisState:
    return AnalysisState(
        dataset_name=dataset_name,
        dataset_schema={},
        cleaning_actions_taken=[],
        findings=[],
        charts_generated=[],
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
        "dataset_schema": state["dataset_schema"],
        "cleaning_actions_taken": state["cleaning_actions_taken"],
        "findings": state["findings"][-max_findings:],
        "charts_generated": [
            {"chart_type": c["chart_type"], "question": c["question"]}
            for c in state["charts_generated"][-max_charts:]
        ],
    }
