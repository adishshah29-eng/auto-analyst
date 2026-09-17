"""Planner agent: the first of the four agents (Planner -> Executor ->
Critic -> Synthesizer). Decides WHAT to do — in plain English, no code —
before the Executor decides HOW.

Splitting "decide what's worth doing" from "write the code for it" gives
each LLM call one job instead of two, and gives the human-in-the-loop gate
something worth reading: a plan is legible to a non-technical reviewer in
a way generated pandas code isn't (see README "Human-in-the-Loop").
"""

from __future__ import annotations

import json

from agent.llm import CostTracker, call_llm, extract_json
from agent.stages.common import format_data_block
from agent.state import AnalysisState, PlannedStep

_SUGGEST_SYSTEM = """You are a planning agent for a data analysis pipeline. Given only a dataset's
schema and aggregated statistics (never raw rows), propose concrete questions THIS dataset could
answer — the kind a person who owns this data would actually want answered.

Rules:
- Name real columns from the schema. "Which city has the highest average Final_Fare?" is a good
  question; "What are the trends?" is not.
- Vary the shape: comparisons across a category, relationships between two numerics, distribution
  or outlier questions, time trends if there's a date column. Don't propose five variants of the
  same question.
- Only propose what this schema can actually support (no time-trend question without a date
  column, no group-by question without a categorical column).

Respond with a single ```json code block: {"questions": ["...", "...", ...]}
Give 4-6 questions, each one sentence.
"""

_SYSTEM = """You are a planning agent for a data analysis pipeline. You do not write code — you
decide what should happen, in plain English, based only on a dataset's schema and aggregated
statistics (never raw rows).

Produce two lists:
1. "cleaning_steps": what to do about nulls, dtype mismatches, and duplicates — only propose
   steps this schema actually needs (e.g. don't propose deduplication if there's no evidence of
   duplicates; don't propose imputation for a column with 0% nulls). Each step is one plain
   sentence describing the action and why (e.g. "Impute missing values in 'Age' with the median,
   since 12% are missing and the column is otherwise numeric.").
2. "exploration_steps": what's worth analyzing given THIS schema — distributions, correlations,
   outliers, group-bys — chosen because they apply to this schema, not a fixed checklist run
   identically every time (e.g. skip correlation analysis if there's only one numeric column;
   skip group-bys if there's no categorical column). Each step is one plain sentence naming the
   specific columns and what to check.

If the user stated what they want to know, the exploration steps exist to ANSWER THAT — make every
step serve their question, and don't pad the list with unrelated analyses just because the schema
would allow them. Add an unrequested step only if it's needed to make sense of their answer (e.g.
checking an outlier that would otherwise distort the average they asked about). Cleaning steps are
unaffected by their question — clean what the data needs either way.

Respond with a single ```json code block: {"cleaning_steps": ["...", ...], "exploration_steps": ["...", ...]}
Aim for 1-4 cleaning steps and 3-6 exploration steps. Fewer, well-justified steps beat padding
the list.
"""

_SCHEMA_LABEL = "dataset schema (dtypes, null %, cardinality, aggregated stats only, no raw rows)"

_USER_TEMPLATE = """{profile_json}
"""

_USER_TEMPLATE_WITH_GOAL = """What the user wants to know from this dataset:
{user_goal}

{profile_json}
"""


def suggest_questions(state: AnalysisState, tracker: CostTracker, model: str) -> list[str]:
    """Schema-aware starting points for the human-in-the-loop intent gate —
    what this specific dataset could answer, so the human picks from real
    options instead of facing an empty text box."""
    profile_json = format_data_block(_SCHEMA_LABEL, state["dataset_schema"], max_chars=6000)
    resp = call_llm(
        system=_SUGGEST_SYSTEM,
        user_message=_USER_TEMPLATE.format(profile_json=profile_json),
        tracker=tracker,
        model=model,
        max_tokens=512,
        stage="suggest",
    )
    try:
        parsed = extract_json(resp.text)
    except (json.JSONDecodeError, ValueError):
        # No suggestions just means the human gets a blank box instead of
        # prefilled options — degrade to that, don't fail the run.
        return []
    return [q.strip() for q in (parsed.get("questions") or []) if isinstance(q, str) and q.strip()]


def plan(state: AnalysisState, tracker: CostTracker, model: str) -> list[PlannedStep]:
    profile_json = format_data_block(_SCHEMA_LABEL, state["dataset_schema"], max_chars=6000)
    user_goal = state.get("user_goal", "").strip()
    if user_goal:
        user_message = _USER_TEMPLATE_WITH_GOAL.format(user_goal=user_goal[:2000], profile_json=profile_json)
    else:
        user_message = _USER_TEMPLATE.format(profile_json=profile_json)

    resp = call_llm(system=_SYSTEM, user_message=user_message, tracker=tracker, model=model, max_tokens=1024, stage="plan")

    try:
        parsed = extract_json(resp.text)
    except (json.JSONDecodeError, ValueError):
        # A planner that fails to parse shouldn't crash the whole run — an
        # empty plan just means the Executor stages fall back to deciding
        # for themselves (their pre-Planner behavior), not a hard failure.
        return []

    steps: list[PlannedStep] = []
    for desc in parsed.get("cleaning_steps", []) or []:
        if isinstance(desc, str) and desc.strip():
            steps.append(PlannedStep(stage="clean", description=desc.strip()))
    for desc in parsed.get("exploration_steps", []) or []:
        if isinstance(desc, str) and desc.strip():
            steps.append(PlannedStep(stage="explore", description=desc.strip()))
    return steps


def cleaning_steps(plan_list: list[PlannedStep]) -> list[str]:
    return [s["description"] for s in plan_list if s["stage"] == "clean"]


def exploration_steps(plan_list: list[PlannedStep]) -> list[str]:
    return [s["description"] for s in plan_list if s["stage"] == "explore"]
