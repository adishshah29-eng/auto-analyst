"""Orchestrates the full pipeline across four agents:

    Planner -> Executor (Clean, Explore) -> Critic -> Executor (Chart) -> Synthesizer

Split into two entry points so a caller can pause for human review between
the Planner and the Executor:

- `plan_analysis()` loads the dataset, profiles it, and asks the Planner
  for a plan (plain English, no code yet). Returns a checkpoint the caller
  can inspect — and the human-in-the-loop gate in app.py edits
  `state["plan"]` here, before anything below has run.
- `execute_analysis()` takes that checkpoint (with whatever plan ended up
  approved) and runs Clean -> Explore -> Critic -> Chart -> Synthesize.
- `run_analysis()` is the convenience wrapper for non-interactive callers
  (eval harness, MCP server, tests): plan then immediately execute with no
  human in the loop.

See README "Pipeline" for what each stage does and why synthesis is kept
separate from the analysis stages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from agent.agents import critic, planner
from agent.llm import DEFAULT_MODEL, BudgetExceededError, CostTracker
from agent.stages import chart, clean, explore, load_profile, synthesize
from agent.state import AnalysisState, StageTimer, new_state

ProgressCallback = Callable[[str, str], None]  # (stage_name, message) -> None


@dataclass
class PlanCheckpoint:
    """Returned by plan_analysis(). Holds everything execute_analysis()
    needs to resume — the caller may freely edit checkpoint.state["plan"]
    (add/remove/reword steps) before passing it to execute_analysis()."""

    state: AnalysisState
    df: pd.DataFrame
    tracker: CostTracker
    chart_dir: str
    model: str


@dataclass
class AnalysisRunResult:
    state: AnalysisState
    cleaned_df: pd.DataFrame
    tracker: CostTracker
    stopped_early: str | None = None  # reason, if a budget ceiling cut the run short


def plan_analysis(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
) -> PlanCheckpoint:
    os.makedirs(chart_dir, exist_ok=True)
    name = dataset_name or os.path.basename(dataset_path)
    state = new_state(name)
    tracker = CostTracker(budget_usd=budget_usd)

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    df = load_profile.load_dataset(dataset_path)

    with StageTimer(state, "load_profile"):
        notify("load_profile", f"Loaded {df.shape[0]} rows x {df.shape[1]} cols. Profiling schema...")
        load_profile.run(state, df)
    notify("load_profile", "Profile complete.")

    with StageTimer(state, "plan"):
        notify("plan", "Planner is deciding what cleaning and analysis steps to take...")
        state["plan"] = planner.plan(state, tracker, model)
    notify(
        "plan",
        f"Plan ready: {len(planner.cleaning_steps(state['plan']))} cleaning step(s), "
        f"{len(planner.exploration_steps(state['plan']))} exploration step(s).",
    )

    return PlanCheckpoint(state=state, df=df, tracker=tracker, chart_dir=chart_dir, model=model)


def execute_analysis(
    checkpoint: PlanCheckpoint,
    on_progress: ProgressCallback | None = None,
    run_critic: bool = True,
    judge_model: str | None = None,
) -> AnalysisRunResult:
    state, df, tracker, chart_dir, model = (
        checkpoint.state,
        checkpoint.df,
        checkpoint.tracker,
        checkpoint.chart_dir,
        checkpoint.model,
    )
    # A model judging its own output is weaker evidence than an independent
    # judge — it's more likely to rate its own confident-sounding-but-wrong
    # narrative as fine. Defaults to self-judging (same model throughout)
    # since that's zero extra config for the common case, but callers that
    # care about trustworthy eval numbers (see eval/run_eval.py --judge-model)
    # can point this at a different, stronger model.
    judge_model = judge_model or model

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    cleaning_plan = planner.cleaning_steps(state["plan"])
    exploration_plan = planner.exploration_steps(state["plan"])

    try:
        with StageTimer(state, "clean"):
            notify("clean", "Executor is implementing the approved cleaning steps...")
            df = clean.run(state, df, tracker, model, planned_steps=cleaning_plan)
        notify("clean", f"Cleaning done: {len(state['cleaning_actions_taken'])} action(s) taken.")

        with StageTimer(state, "explore"):
            notify("explore", "Executor is computing the planned analyses...")
            explore.run(state, df, tracker, model, planned_steps=exploration_plan)
        notify("explore", f"Exploration done: {len(state['findings'])} finding(s).")

        if run_critic and state["findings"]:
            with StageTimer(state, "critic"):
                notify("critic", "Critic is reviewing findings for quality before charting...")
                review = critic.review_findings(state, tracker, model)
            notify("critic", f"Critic kept {review['kept']}, dropped {review['dropped']} finding(s).")

        with StageTimer(state, "chart"):
            notify("chart", "Executor is selecting and generating charts...")
            chart.run(state, df, tracker, model, chart_dir)
        notify("chart", f"Charting done: {len(state['charts_generated'])} chart(s) generated.")

        with StageTimer(state, "synthesize"):
            notify("synthesize", "Synthesizer is writing the final insight summary...")
            synthesize.run(state, tracker, model)
        notify("synthesize", "Summary complete.")

        if run_critic and state["narrative_summary"]:
            with StageTimer(state, "judge"):
                notify("judge", "Critic is scoring the narrative for groundedness and relevance...")
                state["narrative_review"] = critic.review_narrative(state, tracker, judge_model)
            nr = state["narrative_review"]
            notify(
                "judge",
                f"Judge scores — grounded: {nr['grounded_score']}/5, non-obvious: {nr['non_obvious_score']}/5.",
            )

    except BudgetExceededError as e:
        notify("budget", str(e))
        return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker, stopped_early=str(e))

    return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker)


def run_analysis(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
    run_critic: bool = True,
    judge_model: str | None = None,
) -> AnalysisRunResult:
    """Convenience wrapper for non-interactive callers (eval harness, MCP
    server, tests): plan then immediately execute with no human review."""
    checkpoint = plan_analysis(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        model=model,
        budget_usd=budget_usd,
        chart_dir=chart_dir,
        on_progress=on_progress,
    )
    return execute_analysis(checkpoint, on_progress=on_progress, run_critic=run_critic, judge_model=judge_model)
