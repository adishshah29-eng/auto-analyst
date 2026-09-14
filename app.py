"""Streamlit frontend: upload a dataset, watch the agent work through each
pipeline stage live, see the charts it chose to make, and read the final
insight summary."""

from __future__ import annotations

import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from agent.llm import DEFAULT_MODEL, infer_provider
from agent.loop import run_analysis

load_dotenv()

# Streamlit Community Cloud's secrets UI populates st.secrets, not
# os.environ/.env — mirror any secrets it has into os.environ so the rest
# of the app (agent/llm.py reads plain env vars) works unchanged locally
# and on Cloud. A no-op wherever secrets.toml doesn't exist (e.g. local
# dev without one), since st.secrets then just has nothing to iterate.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except FileNotFoundError:
    pass

st.set_page_config(page_title="Autonomous Data Analysis Agent", layout="wide")

STAGE_LABELS = {
    "load_profile": "1. Load & Profile",
    "clean": "2. Clean",
    "explore": "3. Exploratory Analysis",
    "chart": "4. Chart Generation",
    "synthesize": "5. Insight Synthesis",
    "budget": "Budget ceiling",
}

st.title("Autonomous Data Analysis Agent")
st.caption(
    "Upload any CSV. The agent writes and executes real pandas/matplotlib code in a sandbox "
    "to clean, analyze, and chart it — no fixed tools, no hardcoded assumptions about your columns."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL)
    budget = st.number_input("Budget ceiling (USD, 0 = no limit)", min_value=0.0, value=0.50, step=0.10)
    st.markdown("---")
    st.markdown(
        "**Security note:** raw cell values are never sent to the reasoning model — only "
        "the schema and aggregated statistics are. Generated code runs in a restricted, "
        "timeout- and memory-limited sandbox against a *copy* of your data."
    )
    provider = infer_provider(model)
    key_env_var = "GOOGLE_API_KEY" if provider == "google" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_env_var):
        st.warning(f"{key_env_var} is not set for provider '{provider}'. Copy .env.example to .env and add your key.")

uploaded = st.file_uploader("Upload a dataset", type=["csv", "json", "xlsx", "xls"])

if uploaded is not None:
    if st.button("Run analysis", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        chart_dir = tempfile.mkdtemp(prefix="autoanalyst_charts_")

        progress_area = st.container()
        progress_lines: list[str] = []

        def on_progress(stage: str, msg: str) -> None:
            label = STAGE_LABELS.get(stage, stage)
            progress_lines.append(f"**{label}** — {msg}")
            progress_area.markdown("\n\n".join(progress_lines))

        start = time.monotonic()
        with st.spinner("Agent is working through the pipeline..."):
            try:
                result = run_analysis(
                    dataset_path=tmp_path,
                    dataset_name=uploaded.name,
                    model=model,
                    budget_usd=(budget if budget > 0 else None),
                    chart_dir=chart_dir,
                    on_progress=on_progress,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        elapsed = time.monotonic() - start

        state = result.state

        if result.stopped_early:
            st.warning(f"Run stopped early: {result.stopped_early}")

        st.markdown("## Insight Summary")
        st.write(state["narrative_summary"] or "_No summary was produced._")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Latency", f"{elapsed:.1f}s")
        col2.metric("Est. cost", f"${result.tracker.total_cost_usd:.4f}")
        col3.metric("Findings", len(state["findings"]))
        col4.metric("Charts", len(state["charts_generated"]))

        st.markdown("## Charts")
        if state["charts_generated"]:
            cols = st.columns(2)
            for i, c in enumerate(state["charts_generated"]):
                with cols[i % 2]:
                    st.image(c["path"], caption=c["question"] or c["chart_type"])
        else:
            st.write("_No charts were generated._")

        with st.expander("Dataset profile"):
            st.json(state["dataset_schema"])

        with st.expander(f"Cleaning actions ({len(state['cleaning_actions_taken'])})"):
            for a in state["cleaning_actions_taken"]:
                st.write(f"- {a}")

        with st.expander(f"Findings ({len(state['findings'])})"):
            for f in state["findings"]:
                st.write(f"**[{f['kind']}]** {f['description']}")
                if f["stats"]:
                    st.json(f["stats"])

        with st.expander(f"Generated code / execution log ({len(state['code_history'])} steps)"):
            for step in state["code_history"]:
                status = "retry succeeded" if step["retried"] and step["success"] else (
                    "ok" if step["success"] else "failed"
                )
                st.markdown(f"**{step['stage']}** — {status}")
                st.code(step["code"], language="python")
                if step["error"]:
                    st.code(step["error"], language="text")

        st.markdown("## Cost & timing")
        st.json(
            {
                "total_cost_usd": round(result.tracker.total_cost_usd, 5),
                "total_input_tokens": result.tracker.total_input_tokens,
                "total_output_tokens": result.tracker.total_output_tokens,
                "stage_timings_s": {k: round(v, 2) for k, v in state["stage_timings_s"].items()},
            }
        )
else:
    st.info("Upload a CSV (or JSON/Excel) file to begin.")
