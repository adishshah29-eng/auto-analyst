"""MCP server exposing this agent as a single tool, `analyze_dataset`, so
any MCP-aware client (Claude Desktop, Claude Code, another agent) can call
it directly — no browser, no Streamlit UI. Reuses agent/loop.py exactly as
app.py does; this file is a thin adapter, not a second implementation of
the pipeline.

Run directly for local testing:
    mcp dev mcp_server.py

Register with Claude Desktop by adding to claude_desktop_config.json:
    {
      "mcpServers": {
        "auto-analyst": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["/absolute/path/to/mcp_server.py"]
        }
      }
    }
See README "MCP Server" for the full walkthrough.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # before any agent import — see app.py's fix for why this order matters

from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

from agent.llm import DEFAULT_MODEL  # noqa: E402
from agent.loop import run_analysis  # noqa: E402

server = FastMCP(
    "auto-analyst",
    instructions=(
        "Analyzes a CSV/JSON/Excel dataset autonomously: profiles its schema, cleans it, runs "
        "exploratory analysis, generates appropriate charts, and writes a plain-English insight "
        "summary — using a CodeAct agent that writes and executes real pandas/matplotlib code in "
        "a sandbox, not fixed tools. Call analyze_dataset with a local file path."
    ),
)


@server.tool()
def analyze_dataset(
    file_path: str, question: str = "", model: str = DEFAULT_MODEL, budget_usd: float = 1.0
) -> list:
    # Untyped `list` return, deliberately: annotating this as list[str |
    # Image] makes FastMCP try to build a pydantic output schema from the
    # annotation, and Image (a content-conversion marker, not a schema
    # type) can't produce one — caught by an actual MCP client connecting
    # over stdio, which failed with "Connection closed" at initialize()
    # because the server crashed constructing its tool list on startup.
    """Run the full autonomous analysis pipeline (Planner -> Executor ->
    Critic -> Synthesizer) against a local dataset file and return the
    insight summary, key findings, and generated charts.

    Args:
        file_path: absolute path to a local .csv, .json, .xlsx, or .xls file.
        question: what you want to know from this data, in plain English
            (e.g. "which city has the highest average fare, and is it
            rising?"). Steers which analyses run, which charts get made,
            and makes the summary answer this directly. Omit it to let the
            agent decide what's interesting on its own.
        model: which model to use (any Claude or Gemini model id). Defaults
            to the server's configured ANALYSIS_MODEL.
        budget_usd: stop the run early if estimated LLM cost exceeds this.
    """
    if not os.path.isfile(file_path):
        return [f"Error: '{file_path}' is not a file the server can read."]

    result = run_analysis(
        dataset_path=file_path, model=model, budget_usd=budget_usd, user_goal=question
    )
    state = result.state

    lines = [f"# Analysis of {state['dataset_name']}", ""]
    if state["user_goal"]:
        lines += [f"**Question asked:** {state['user_goal']}", ""]

    if result.stopped_early:
        lines.append(f"**Run stopped early:** {result.stopped_early}")
        lines.append("")

    lines.append(f"**Rows x Cols:** {state['dataset_schema'].get('n_rows')} x {state['dataset_schema'].get('n_cols')}")
    lines.append(f"**Cost:** ${result.tracker.total_cost_usd:.4f}  |  **Findings:** {len(state['findings'])}  |  **Charts:** {len(state['charts_generated'])}")
    lines.append("")

    lines.append("## Insight Summary")
    lines.append(state["narrative_summary"] or "_No summary was produced._")
    lines.append("")

    nr = state["narrative_review"]
    if nr and nr["grounded_score"] is not None:
        lines.append(
            f"_Critic's judge score — grounded: {nr['grounded_score']}/5, "
            f"non-obvious: {nr['non_obvious_score']}/5, actionable: {nr['actionable']}._"
        )
        lines.append("")

    if state["cleaning_actions_taken"]:
        lines.append("## Cleaning Actions")
        lines.extend(f"- {a}" for a in state["cleaning_actions_taken"])
        lines.append("")

    if state["findings"]:
        lines.append("## Findings")
        for f in state["findings"]:
            lines.append(f"- **[{f['kind']}]** {f['description']}")
        lines.append("")

    content: list[str | Image] = ["\n".join(lines)]
    for c in state["charts_generated"]:
        # Charts are interactive Plotly HTML in the app (see app.py / README
        # "Interactive Charts") — an MCP client can't render live JS, so
        # this path needs an actual raster image. `static_path` is only
        # populated when the optional `kaleido` package was available at
        # chart-generation time; when it wasn't, fall back to a text note
        # rather than silently dropping the chart.
        if c.get("static_path") and os.path.exists(c["static_path"]):
            content.append(Image(path=c["static_path"]))
        elif os.path.exists(c["path"]):
            content.append(
                f"_Chart \"{c['question'] or c['chart_type']}\" is interactive-only "
                f"(install the optional `kaleido` package for a static image here) "
                f"— saved to `{c['path']}`, open it in a browser to view._"
            )

    return content


if __name__ == "__main__":
    server.run()
