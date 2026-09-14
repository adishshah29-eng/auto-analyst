"""Orchestrates the full pipeline: Load & Profile -> Clean -> Explore ->
Chart -> Synthesize. See README "Pipeline" for what each stage does and
why synthesis is kept separate from the analysis stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from agent.llm import DEFAULT_MODEL, BudgetExceededError, CostTracker
from agent.stages import chart, clean, explore, load_profile, synthesize
from agent.state import AnalysisState, StageTimer, new_state

ProgressCallback = Callable[[str, str], None]  # (stage_name, message) -> None


@dataclass
class AnalysisRunResult:
    state: AnalysisState
    cleaned_df: pd.DataFrame
    tracker: CostTracker
    stopped_early: str | None = None  # reason, if a budget ceiling cut the run short


def run_analysis(
    dataset_path: str,
    dataset_name: str | None = None,
    model: str = DEFAULT_MODEL,
    budget_usd: float | None = None,
    chart_dir: str = "outputs/charts",
    on_progress: ProgressCallback | None = None,
) -> AnalysisRunResult:
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

    try:
        with StageTimer(state, "clean"):
            notify("clean", "Agent is writing a cleaning plan...")
            df = clean.run(state, df, tracker, model)
        notify("clean", f"Cleaning done: {len(state['cleaning_actions_taken'])} action(s) taken.")

        with StageTimer(state, "explore"):
            notify("explore", "Agent is exploring distributions, correlations, outliers...")
            explore.run(state, df, tracker, model)
        notify("explore", f"Exploration done: {len(state['findings'])} finding(s).")

        with StageTimer(state, "chart"):
            notify("chart", "Agent is selecting and generating charts...")
            chart.run(state, df, tracker, model, chart_dir)
        notify("chart", f"Charting done: {len(state['charts_generated'])} chart(s) generated.")

        with StageTimer(state, "synthesize"):
            notify("synthesize", "Writing the final insight summary...")
            synthesize.run(state, tracker, model)
        notify("synthesize", "Summary complete.")

    except BudgetExceededError as e:
        notify("budget", str(e))
        return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker, stopped_early=str(e))

    return AnalysisRunResult(state=state, cleaned_df=df, tracker=tracker)
