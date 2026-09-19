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
- `ask_followup()` is a second (or third...) question against the SAME
  already-cleaned dataset from a prior `execute_analysis()` call — skips
  Load/Clean entirely and re-enters at Explore, so asking something else
  doesn't mean re-uploading or re-cleaning. Shares the actual
  Explore->Critic->Chart->Synthesize->Critic tail with `execute_analysis()`
  via `_run_explore_through_judge()` — the two paths only ever differ in
  what happens *before* that point, never in what happens after, so a fix
  or a hardening change to that tail (the significance gate, the
  never-drop-a-caveated-finding policy, tracing) applies to both without
  anyone having to remember to port it.

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
from agent.tracing import RunTracer

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


def profile_and_suggest(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
    suggest: bool = True,
) -> PlanCheckpoint:
    """First half of planning: load, profile, and ask the Planner what
    questions this dataset could answer. Returns before any plan exists —
    the intent gate in app.py shows those suggestions, takes the human's
    answer, and passes it to make_plan().

    `suggest=False` skips the suggestion call entirely: it only exists to
    populate a human-facing picker, so generating it for a non-interactive
    caller (eval, MCP, tests) is a wasted LLM call and wasted latency."""
    os.makedirs(chart_dir, exist_ok=True)
    name = dataset_name or os.path.basename(dataset_path)
    tracer = RunTracer(dataset_name=name)
    state = new_state(name, run_id=tracer.run_id)
    tracker = CostTracker(budget_usd=budget_usd, tracer=tracer)

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    df = load_profile.load_dataset(dataset_path)

    with StageTimer(state, "load_profile"):
        tracer.log_stage_boundary("load_profile", "start")
        notify("load_profile", f"Loaded {df.shape[0]} rows x {df.shape[1]} cols. Profiling schema...")
        load_profile.run(state, df)
    notify("load_profile", "Profile complete.")
    tracer.log_stage_boundary("load_profile", "end")

    if suggest:
        with StageTimer(state, "suggest"):
            tracer.log_stage_boundary("suggest", "start")
            notify("suggest", "Planner is working out what this dataset can answer...")
            state["suggested_questions"] = planner.suggest_questions(state, tracker, model)
        notify("suggest", f"{len(state['suggested_questions'])} question(s) suggested.")
        tracer.log_stage_boundary("suggest", "end")

    return PlanCheckpoint(state=state, df=df, tracker=tracker, chart_dir=chart_dir, model=model)


def make_plan(
    checkpoint: PlanCheckpoint,
    user_goal: str = "",
    on_progress: ProgressCallback | None = None,
) -> PlanCheckpoint:
    """Second half of planning: turn the human's stated goal (empty string =
    "just analyze it") plus the schema into a concrete plan. Mutates and
    returns the same checkpoint."""
    state, tracker, model = checkpoint.state, checkpoint.tracker, checkpoint.model

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    state["user_goal"] = user_goal.strip()
    tracer = tracker.tracer

    with StageTimer(state, "plan"):
        if tracer is not None:
            tracer.log_stage_boundary("plan", "start")
        notify(
            "plan",
            "Planner is building a plan to answer your question..."
            if state["user_goal"]
            else "Planner is deciding what cleaning and analysis steps to take...",
        )
        state["plan"] = planner.plan(state, tracker, model)
    notify(
        "plan",
        f"Plan ready: {len(planner.cleaning_steps(state['plan']))} cleaning step(s), "
        f"{len(planner.exploration_steps(state['plan']))} exploration step(s).",
    )
    if tracer is not None:
        tracer.log_stage_boundary("plan", "end")

    return checkpoint


def plan_analysis(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
    user_goal: str = "",
) -> PlanCheckpoint:
    """Both halves in one call, for non-interactive callers that already
    know the goal (or have none): eval harness, MCP server, tests."""
    checkpoint = profile_and_suggest(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        model=model,
        budget_usd=budget_usd,
        chart_dir=chart_dir,
        on_progress=on_progress,
        suggest=False,  # nobody reads suggestions on this path — don't pay for them
    )
    return make_plan(checkpoint, user_goal=user_goal, on_progress=on_progress)


def _run_explore_through_judge(
    state: AnalysisState,
    df: pd.DataFrame,
    tracker: CostTracker,
    model: str,
    chart_dir: str,
    exploration_plan: list[str],
    on_progress: ProgressCallback | None,
    run_critic: bool,
    judge_model: str,
) -> AnalysisRunResult:
    """The shared tail: Explore -> Critic(findings) -> Chart -> Synthesize
    -> Critic(narrative). `execute_analysis()` reaches this after Clean;
    `ask_followup()` reaches it after re-profiling an already-cleaned
    dataframe — see module docstring for why this is factored out rather
    than duplicated."""
    tracer = tracker.tracer

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    def trace(stage: str, event: str) -> None:
        if tracer is not None:
            tracer.log_stage_boundary(stage, event)

    try:
        with StageTimer(state, "explore"):
            trace("explore", "start")
            notify("explore", "Executor is computing the planned analyses...")
            explore.run(state, df, tracker, model, planned_steps=exploration_plan)
        notify("explore", f"Exploration done: {len(state['findings'])} finding(s).")
        trace("explore", "end")

        if run_critic and state["findings"]:
            with StageTimer(state, "critic"):
                trace("critic", "start")
                notify("critic", "Critic is reviewing findings for quality before charting...")
                review = critic.review_findings(state, tracker, model)
            notify("critic", f"Critic kept {review['kept']}, dropped {review['dropped']} finding(s).")
            trace("critic", "end")

        with StageTimer(state, "chart"):
            trace("chart", "start")
            notify("chart", "Executor is selecting and generating charts...")
            chart.run(state, df, tracker, model, chart_dir)
        notify("chart", f"Charting done: {len(state['charts_generated'])} chart(s) generated.")
        trace("chart", "end")

        with StageTimer(state, "synthesize"):
            trace("synthesize", "start")
            notify("synthesize", "Synthesizer is writing the final insight summary...")
            synthesize.run(state, tracker, model)
        notify("synthesize", "Summary complete.")
        trace("synthesize", "end")

        if run_critic and state["narrative_summary"]:
            with StageTimer(state, "judge"):
                trace("judge", "start")
                notify("judge", "Critic is scoring the narrative for groundedness and relevance...")
                state["narrative_review"] = critic.review_narrative(state, tracker, judge_model)
            nr = state["narrative_review"]
            notify(
                "judge",
                f"Judge scores — grounded: {nr['grounded_score']}/5, non-obvious: {nr['non_obvious_score']}/5.",
            )
            trace("judge", "end")

    except BudgetExceededError as e:
        notify("budget", str(e))
        return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker, stopped_early=str(e))

    return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker)


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
    tracer = tracker.tracer

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    def trace(stage: str, event: str) -> None:
        if tracer is not None:
            tracer.log_stage_boundary(stage, event)

    cleaning_plan = planner.cleaning_steps(state["plan"])
    exploration_plan = planner.exploration_steps(state["plan"])

    try:
        with StageTimer(state, "clean"):
            trace("clean", "start")
            notify("clean", "Executor is implementing the approved cleaning steps...")
            df = clean.run(state, df, tracker, model, planned_steps=cleaning_plan)
        notify("clean", f"Cleaning done: {len(state['cleaning_actions_taken'])} action(s) taken.")
        trace("clean", "end")
    except BudgetExceededError as e:
        notify("budget", str(e))
        return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker, stopped_early=str(e))

    return _run_explore_through_judge(
        state, df, tracker, model, chart_dir, exploration_plan, on_progress, run_critic, judge_model
    )


def ask_followup(
    checkpoint: PlanCheckpoint,
    cleaned_df: pd.DataFrame,
    user_goal: str,
    on_progress: ProgressCallback | None = None,
    run_critic: bool = True,
    judge_model: str | None = None,
) -> AnalysisRunResult:
    """A second (or third...) question against the SAME already-cleaned
    dataset from a prior `execute_analysis()` call — the interactive path a
    real analysis session actually needs: ask something, read the answer,
    ask a follow-up, without re-uploading or re-cleaning.

    Skips Load/Clean entirely. Re-profiles `cleaned_df` (cheap, no LLM —
    agent/stages/load_profile.py) rather than reusing the pre-cleaning
    schema from the first round, so the Planner/Explore/judge see accurate
    stats for the data they're actually querying now (null percentages,
    category top-values, etc. all reflect the cleaning already done, not
    the raw upload). Builds a fresh AnalysisState — its own findings,
    charts, and narrative — rather than appending to the prior one, so
    each question gets its own self-contained answer; the caller (app.py)
    is expected to keep a list of past results for a running Q&A view.

    Reuses `checkpoint.tracker` (so the budget ceiling and running cost
    total span the whole session, not just one question) and its run_id
    (so every question in a session lands in the same trace file)."""
    tracker, model, chart_dir = checkpoint.tracker, checkpoint.model, checkpoint.chart_dir
    dataset_name = checkpoint.state["dataset_name"]
    tracer = tracker.tracer

    state = new_state(dataset_name, run_id=(tracer.run_id if tracer is not None else ""))
    state["user_goal"] = user_goal.strip()

    def notify(stage: str, msg: str) -> None:
        if on_progress:
            on_progress(stage, msg)

    with StageTimer(state, "load_profile"):
        if tracer is not None:
            tracer.log_stage_boundary("load_profile", "start")
        notify("load_profile", "Re-profiling the already-cleaned dataset for your follow-up...")
        load_profile.run(state, cleaned_df)
    notify("load_profile", "Profile complete.")
    if tracer is not None:
        tracer.log_stage_boundary("load_profile", "end")

    with StageTimer(state, "plan"):
        if tracer is not None:
            tracer.log_stage_boundary("plan", "start")
        notify("plan", "Planner is working out how to answer your follow-up...")
        state["plan"] = planner.plan(state, tracker, model)
    exploration_plan = planner.exploration_steps(state["plan"])
    notify("plan", f"Plan ready: {len(exploration_plan)} exploration step(s).")
    if tracer is not None:
        tracer.log_stage_boundary("plan", "end")

    return _run_explore_through_judge(
        state, cleaned_df, tracker, model, chart_dir, exploration_plan, on_progress, run_critic, judge_model or model
    )


def run_analysis(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
    run_critic: bool = True,
    judge_model: str | None = None,
    user_goal: str = "",
) -> AnalysisRunResult:
    """Convenience wrapper for non-interactive callers (eval harness, MCP
    server, tests): plan then immediately execute with no human review.
    `user_goal` is the same steering the UI's intent gate provides — an MCP
    caller passing a question gets the same goal-directed behavior a human
    typing one into the app does."""
    checkpoint = plan_analysis(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        model=model,
        budget_usd=budget_usd,
        chart_dir=chart_dir,
        on_progress=on_progress,
        user_goal=user_goal,
    )
    return execute_analysis(checkpoint, on_progress=on_progress, run_critic=run_critic, judge_model=judge_model)
