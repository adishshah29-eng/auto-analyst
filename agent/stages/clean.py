"""Stage 2: Clean (Executor half of clean). Implements the cleaning steps
the Planner already decided on and a human may have reviewed/edited — this
stage's job is HOW (write correct pandas code), not WHAT (that was the
Planner's job). Falls back to deciding for itself only if no plan was
supplied, so the stage still works standalone (eval scripts, tests, or any
caller that skips planning)."""

from __future__ import annotations

import json

import pandas as pd

from agent.llm import CostTracker
from agent.state import AnalysisState
from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry

_TASK_WITH_PLAN = """Implement EXACTLY the following approved cleaning steps against `df` — do not
add steps beyond this list, and do not skip any of them:

{plan_text}

Reassign the cleaned result to `df`. Also build a Python list of short strings named
`cleaning_actions` describing each action you took, in the same order as the plan (e.g. "Imputed
12 missing values in 'age' with median").

Dataset profile (schema + aggregated stats only, no raw rows), for column names/dtypes only —
the cleaning decisions themselves are already made, use this just to write correct code:
{profile_json}
"""

_TASK_NO_PLAN = """Given the dataset profile below, decide on and write pandas code that cleans `df`:
- Handle missing values sensibly per column (impute, or drop rows/cols only when null_pct is very high).
- Fix obvious dtype mismatches (e.g. a numeric column stored as text).
- Drop exact duplicate rows if any exist.
- Do NOT drop or rename columns unless clearly justified by the profile (e.g. a column that is entirely null).

Reassign the cleaned result to `df`. Also build a Python list of short strings named
`cleaning_actions` describing each action you took (e.g. "Imputed 12 missing values in 'age' with median").
If no cleaning was needed, set `cleaning_actions = []` and leave `df` unchanged.

Dataset profile (schema + aggregated stats only, no raw rows):
{profile_json}
"""


def run(
    state: AnalysisState,
    df: pd.DataFrame,
    tracker: CostTracker,
    model: str,
    planned_steps: list[str] | None = None,
) -> pd.DataFrame:
    profile_json = json.dumps(state["dataset_schema"], default=str)[:6000]

    if planned_steps:
        plan_text = "\n".join(f"- {s}" for s in planned_steps)
        user_prompt = _TASK_WITH_PLAN.format(plan_text=plan_text, profile_json=profile_json)
    else:
        user_prompt = _TASK_NO_PLAN.format(profile_json=profile_json)

    result, _ = _run_with_retry(
        system=SANDBOX_SYSTEM_PREAMBLE,
        user_prompt=user_prompt,
        df=df,
        capture_vars=["df", "cleaning_actions"],
        stage="clean",
        state=state,
        tracker=tracker,
        model=model,
    )

    if not result.success:
        state["cleaning_actions_taken"].append(
            f"Cleaning step failed after retry, proceeding with uncleaned data: {result.error.splitlines()[-1] if result.error else 'unknown error'}"
        )
        return df

    cleaned_df = result.output_vars.get("df", df)
    actions = result.output_vars.get("cleaning_actions", [])
    if isinstance(actions, list):
        state["cleaning_actions_taken"].extend(str(a) for a in actions)
    return cleaned_df if isinstance(cleaned_df, pd.DataFrame) else df
