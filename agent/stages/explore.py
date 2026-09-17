"""Stage 3: Exploratory Analysis (Executor half of explore). Computes the
analyses the Planner already decided were worth running — this stage's job
is HOW (write correct pandas code to check each planned item), not WHAT.
Falls back to deciding for itself if no plan was supplied, so it still
works standalone."""

from __future__ import annotations

import json

import pandas as pd

from agent.llm import CostTracker
from agent.state import AnalysisState
from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry, format_data_block

_SCHEMA_LABEL = "dataset profile (schema + aggregated stats only, no raw rows)"

_TASK_WITH_PLAN = """{goal_block}Compute EXACTLY the following approved analyses against `df` — one finding per
planned item (skip an item only if the code genuinely can't produce it, e.g. a planned correlation
against a column that turned out to be constant):

{plan_text}

Build a Python list of dicts named `findings`, each shaped like:
{{"kind": "distribution" | "correlation" | "outlier" | "groupby" | "other",
  "description": "<one plain-English sentence stating the finding, with the actual numbers>",
  "stats": {{...small dict of the supporting numbers...}}}}
For "groupby", "correlation", and "outlier" findings, include "n" in stats — the number of rows
the SPECIFIC CLAIM rests on, which for a subset is NOT the dataset's total row count:
- groupby: the row count of the one category the finding names. If you write "category X has the
  highest rate", n is len(df[df[col] == "X"]), NOT len(df).
- outlier: how many rows are actually outliers, not how many rows were scanned.
- correlation: the number of paired non-null observations (this one may legitimately equal the
  total row count).
Compute n from the data — do not assume it. A downstream check flags claims resting on too few
rows, and reporting the dataset size for a subgroup defeats it entirely.
Do not put any raw row-level data into `findings` — aggregated numbers only.

Cleaning already applied: {cleaning_actions}

For column names/dtypes only — the analysis decisions themselves are already made, use this just
to write correct code:
{profile_json}
"""

_TASK_NO_PLAN = """Given the cleaned dataset's profile below, explore `df` and surface what's actually
interesting about THIS dataset. Choose analyses appropriate to the columns present, e.g.:
- distribution shape (skew, spread) for numeric columns worth calling out
- correlations between numeric columns, especially ones that are surprising or strong
- outliers (IQR or z-score) in numeric columns
- group-by aggregations across a categorical column and a numeric one, if there are categorical columns

Do not run every possible analysis mechanically — decide which apply given the schema (e.g. skip
correlation analysis if there's only one numeric column; skip group-bys if there's no categorical column).

Build a Python list of dicts named `findings`, each shaped like:
{{"kind": "distribution" | "correlation" | "outlier" | "groupby" | "other",
  "description": "<one plain-English sentence stating the finding, with the actual numbers>",
  "stats": {{...small dict of the supporting numbers...}}}}
For "groupby", "correlation", and "outlier" findings, include "n" in stats — the number of rows
the SPECIFIC CLAIM rests on, which for a subset is NOT the dataset's total row count:
- groupby: the row count of the one category the finding names. If you write "category X has the
  highest rate", n is len(df[df[col] == "X"]), NOT len(df).
- outlier: how many rows are actually outliers, not how many rows were scanned.
- correlation: the number of paired non-null observations (this one may legitimately equal the
  total row count).
Compute n from the data — do not assume it. A downstream check flags claims resting on too few
rows, and reporting the dataset size for a subgroup defeats it entirely.
Only include findings that are actually notable (skip trivial/obvious ones). Aim for 3-8 findings.
Do not put any raw row-level data into `findings` — aggregated numbers only.

Cleaning already applied: {cleaning_actions}

{profile_json}
"""


def run(
    state: AnalysisState,
    df: pd.DataFrame,
    tracker: CostTracker,
    model: str,
    planned_steps: list[str] | None = None,
) -> None:
    profile_json = format_data_block(_SCHEMA_LABEL, state["dataset_schema"], max_chars=6000)
    cleaning_actions = json.dumps(state["cleaning_actions_taken"][-10:])
    user_goal = state.get("user_goal", "").strip()
    goal_block = (
        f"The user wants to know: {user_goal[:2000]}\nEvery finding you produce should help answer that.\n\n"
        if user_goal
        else ""
    )

    if planned_steps:
        plan_text = "\n".join(f"- {s}" for s in planned_steps)
        user_prompt = _TASK_WITH_PLAN.format(
            goal_block=goal_block, plan_text=plan_text, profile_json=profile_json, cleaning_actions=cleaning_actions
        )
    else:
        user_prompt = _TASK_NO_PLAN.format(profile_json=profile_json, cleaning_actions=cleaning_actions)

    result, _ = _run_with_retry(
        system=SANDBOX_SYSTEM_PREAMBLE,
        user_prompt=user_prompt,
        df=df,
        capture_vars=["findings"],
        stage="explore",
        state=state,
        tracker=tracker,
        model=model,
    )

    if not result.success:
        return

    findings = result.output_vars.get("findings", [])
    if isinstance(findings, list):
        for f in findings:
            if isinstance(f, dict) and "description" in f:
                state["findings"].append(
                    {
                        "kind": str(f.get("kind", "other")),
                        "description": str(f.get("description", "")),
                        "stats": f.get("stats", {}) if isinstance(f.get("stats"), dict) else {},
                        "caveat": "",  # set later by agent.agents.significance if the stats warrant it
                    }
                )
