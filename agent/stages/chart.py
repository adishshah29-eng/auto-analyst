"""Stage 4: Chart Generation. Chart type is chosen per finding, not from a
fixed set generated regardless of content — a skewed numeric column gets a
histogram, a flagged correlation gets a scatter plot, a categorical
breakdown gets a bar chart, etc."""

from __future__ import annotations

import json
import uuid

import pandas as pd

from agent.llm import CostTracker
from agent.state import AnalysisState, ChartMeta
from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry, format_data_block

_TASK = """{goal_block}Given the findings below, create the matplotlib chart(s) that best communicate them.
Pick a chart type appropriate to each finding's content — for example:
- a skewed/notable numeric distribution -> histogram
- a flagged correlation between two numeric columns -> scatter plot
- a categorical breakdown or group-by result -> bar chart
- a time-based finding -> line chart
Do not produce a fixed number of charts regardless of content: create one chart per finding that is
actually chart-worthy (skip findings that don't visualize well), typically 2-4 charts total.
Call `plt.figure()` before each chart so each is captured separately. Give each chart a title and
axis labels.

Build a Python list of dicts named `chart_meta`, one entry per `plt.figure()` you created, IN THE
SAME ORDER, shaped like: {{"chart_type": "histogram"|"scatter"|"bar"|"line"|"box"|"other",
"question": "<the specific question this chart answers, one sentence>"}}.

Findings to visualize:
{findings_json}

{profile_json}
"""


def run(state: AnalysisState, df: pd.DataFrame, tracker: CostTracker, model: str, chart_dir: str) -> None:
    profile_json = format_data_block(
        "dataset profile (schema + aggregated stats only, no raw rows)", state["dataset_schema"]
    )
    findings_json = json.dumps(state["findings"], default=str)[:4000]
    user_goal = state.get("user_goal", "").strip()
    goal_block = (
        f"The user asked: {user_goal[:2000]}\n"
        "Chart what answers THAT question first — the chart that most directly shows their answer "
        "comes first, and skip findings that don't help answer it even if they'd visualize nicely.\n\n"
        if user_goal
        else ""
    )
    user_prompt = _TASK.format(goal_block=goal_block, findings_json=findings_json, profile_json=profile_json)

    result, _ = _run_with_retry(
        system=SANDBOX_SYSTEM_PREAMBLE,
        user_prompt=user_prompt,
        df=df,
        capture_vars=["chart_meta"],
        stage="chart",
        state=state,
        tracker=tracker,
        model=model,
        chart_dir=chart_dir,
        chart_prefix=f"chart_{uuid.uuid4().hex[:8]}",
    )

    if not result.success:
        return

    chart_meta = result.output_vars.get("chart_meta", [])
    if not isinstance(chart_meta, list):
        chart_meta = []

    for path, meta in zip(result.chart_paths, chart_meta):
        if not isinstance(meta, dict):
            continue
        state["charts_generated"].append(
            ChartMeta(
                path=path,
                chart_type=str(meta.get("chart_type", "other")),
                question=str(meta.get("question", "")),
            )
        )
    # Charts without matching metadata (LLM produced more figures than
    # descriptions) still get saved to disk but are recorded generically
    # rather than silently dropped.
    for path in result.chart_paths[len(chart_meta):]:
        state["charts_generated"].append(ChartMeta(path=path, chart_type="other", question=""))
