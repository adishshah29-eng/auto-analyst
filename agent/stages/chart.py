"""Stage 4: Chart Generation. Chart type is chosen per finding, not from a
fixed set generated regardless of content — a skewed numeric column gets a
histogram, a flagged correlation gets a scatter plot, a categorical
breakdown gets a bar chart, etc.

Charts are Plotly figures, not matplotlib — they render as genuinely
interactive (hover tooltips, zoom/pan, legend toggling) in the Streamlit
app, not a static picture. See agent/sandbox.py's chart-collection block
for how a `charts` list of figure objects becomes saved HTML files, and
README "Interactive Charts" for the full design."""

from __future__ import annotations

import json
import uuid

import pandas as pd

from agent.llm import CostTracker
from agent.state import AnalysisState, ChartMeta
from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry, format_data_block

_TASK = """{goal_block}Given the findings below, create the Plotly chart(s) that best communicate them.
Pick a chart type appropriate to each finding's content — for example:
- a skewed/notable numeric distribution -> histogram (px.histogram)
- a flagged correlation between two numeric columns -> scatter plot (px.scatter)
- a categorical breakdown or group-by result -> bar chart (px.bar)
- a time-based finding -> line chart (px.line)
Do not produce a fixed number of charts regardless of content: create one chart per finding that is
actually chart-worthy (skip findings that don't visualize well), typically 2-4 charts total.
Build each chart with px (plotly.express) or go (plotly.graph_objects) as a Figure object. Give
each one a title and axis labels via `fig.update_layout(title=..., xaxis_title=..., yaxis_title=...)`.
Append each finished figure to a Python list named `charts`, IN THE SAME ORDER you build them —
this is how figures are captured, there is no separate "show" step.

Build a second Python list of dicts named `chart_meta`, one entry per figure in `charts`, IN THE
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

    def _static_path(i: int) -> str:
        return result.static_chart_paths[i] if i < len(result.static_chart_paths) else ""

    for i, (path, meta) in enumerate(zip(result.chart_paths, chart_meta)):
        if not isinstance(meta, dict):
            continue
        state["charts_generated"].append(
            ChartMeta(
                path=path,
                static_path=_static_path(i),
                chart_type=str(meta.get("chart_type", "other")),
                question=str(meta.get("question", "")),
            )
        )
    # Charts without matching metadata (LLM produced more figures than
    # descriptions) still get saved to disk but are recorded generically
    # rather than silently dropped.
    for i in range(len(chart_meta), len(result.chart_paths)):
        state["charts_generated"].append(
            ChartMeta(path=result.chart_paths[i], static_path=_static_path(i), chart_type="other", question="")
        )
