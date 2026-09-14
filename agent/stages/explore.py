"""Stage 3: Exploratory Analysis. Agent decides what's worth looking at
(distributions, correlations, outliers, group-bys) based on the schema it
profiled and the cleaning it already did — not a fixed checklist run
identically on every dataset."""

from __future__ import annotations

import json

import pandas as pd

from agent.llm import CostTracker
from agent.state import AnalysisState
from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry

_TASK = """Given the cleaned dataset's profile below, explore `df` and surface what's actually
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
Only include findings that are actually notable (skip trivial/obvious ones). Aim for 3-8 findings.
Do not put any raw row-level data into `findings` — aggregated numbers only.

Cleaning already applied: {cleaning_actions}

Dataset profile (schema + aggregated stats only, no raw rows):
{profile_json}
"""


def run(state: AnalysisState, df: pd.DataFrame, tracker: CostTracker, model: str) -> None:
    profile_json = json.dumps(state["dataset_schema"], default=str)[:6000]
    user_prompt = _TASK.format(
        profile_json=profile_json,
        cleaning_actions=json.dumps(state["cleaning_actions_taken"][-10:]),
    )

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
                    }
                )
