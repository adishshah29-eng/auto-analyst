"""Streamlit frontend: upload a dataset, review the Planner's proposed
cleaning steps (human-in-the-loop gate), then watch Executor -> Critic ->
Synthesizer run, see the charts, and read the final insight summary."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

# Must run before any `from agent...` import: agent/llm.py reads
# ANALYSIS_MODEL from the environment at *import* time to compute
# DEFAULT_MODEL, so calling load_dotenv() after those imports (as this file
# did until this fix) meant .env's ANALYSIS_MODEL was silently ignored and
# the sidebar always showed the hardcoded "claude-sonnet-5" fallback —
# caught by an actual Playwright browser run, not just an HTTP-200 boot
# check, which never exercises what the sidebar displays.
load_dotenv()

from agent.agents import planner  # noqa: E402
from agent.llm import DEFAULT_MODEL, infer_provider  # noqa: E402
from agent.loop import execute_analysis, plan_analysis  # noqa: E402
from agent.state import PlannedStep  # noqa: E402


def _running_commit_short() -> str:
    """Best-effort git SHA of the code the server is actually running.
    Fails silently to 'unknown' when git isn't available or when the deploy
    doesn't ship .git — the value is diagnostic, never critical."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
    except Exception:
        return "unknown"

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
    "plan": "2. Planner",
    "clean": "3. Executor — Clean",
    "explore": "4. Executor — Explore",
    "critic": "5. Critic (findings)",
    "chart": "6. Executor — Chart",
    "synthesize": "7. Synthesizer",
    "judge": "8. Critic (narrative)",
    "budget": "Budget ceiling",
}

st.title("Autonomous Data Analysis Agent")
st.caption(
    "Upload any CSV. Four agents — Planner, Executor, Critic, Synthesizer — write and execute "
    "real pandas/matplotlib code in a sandbox to clean, analyze, and chart it. No fixed tools, "
    "no hardcoded assumptions about your columns."
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
    st.caption(f"Running commit: `{_running_commit_short()}` — compare with the latest on GitHub to check whether the deploy has picked up your last push.")

if "checkpoint" not in st.session_state:
    st.session_state.checkpoint = None  # PlanCheckpoint, set once planning completes
if "result" not in st.session_state:
    st.session_state.result = None  # AnalysisRunResult, set once execution completes
if "elapsed_plan" not in st.session_state:
    st.session_state.elapsed_plan = 0.0
if "elapsed_exec" not in st.session_state:
    st.session_state.elapsed_exec = 0.0


def _make_progress_renderer():
    area = st.empty()
    lines: list[str] = []

    def on_progress(stage: str, msg: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        lines.append(f"**{label}** — {msg}")
        area.markdown("\n\n".join(lines))

    return on_progress


def _reset() -> None:
    st.session_state.checkpoint = None
    st.session_state.result = None
    st.session_state.elapsed_plan = 0.0
    st.session_state.elapsed_exec = 0.0


uploaded = st.file_uploader("Upload a dataset", type=["csv", "json", "xlsx", "xls"])

if uploaded is None:
    _reset()
    st.info("Upload a CSV (or JSON/Excel) file to begin.")

elif st.session_state.result is not None:
    # ----- Stage 3: results -----
    result = st.session_state.result
    state = result.state

    if st.button("Start a new analysis"):
        _reset()
        st.rerun()

    if result.stopped_early:
        st.warning(f"Run stopped early: {result.stopped_early}")

    st.markdown("## Insight Summary")
    st.write(state["narrative_summary"] or "_No summary was produced._")

    nr = state["narrative_review"]
    if nr and nr["grounded_score"] is not None:
        st.caption(
            f"Critic's judge score — grounded: {nr['grounded_score']}/5, "
            f"non-obvious: {nr['non_obvious_score']}/5, actionable: {nr['actionable']}. "
            f"_{nr['reasoning']}_"
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{st.session_state.elapsed_plan + st.session_state.elapsed_exec:.1f}s")
    col2.metric("Est. cost", f"${result.tracker.total_cost_usd:.4f}")
    col3.metric("Findings", len(state["findings"]))
    col4.metric("Charts", len(state["charts_generated"]))

    if state["critic_review"] and state["critic_review"]["dropped"]:
        with st.expander(f"Critic dropped {state['critic_review']['dropped']} finding(s) before charting"):
            for reason in state["critic_review"]["reasons"]:
                st.write(f"- {reason}")

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

elif st.session_state.checkpoint is not None:
    # ----- Stage 2: human-in-the-loop plan review -----
    checkpoint = st.session_state.checkpoint
    state = checkpoint.state

    st.markdown("## Planner's proposed plan")
    st.caption(
        "The Planner decided what to do — in plain English, no code yet. Review and edit the "
        "cleaning steps below before the Executor writes and runs any code."
    )

    clean_steps = planner.cleaning_steps(state["plan"])
    explore_steps = planner.exploration_steps(state["plan"])

    st.markdown("**Proposed cleaning steps** (edit freely — one per line, delete a line to skip it):")
    edited_text = st.text_area(
        "Cleaning steps", value="\n".join(clean_steps), height=150, label_visibility="collapsed"
    )

    with st.expander(f"Planned exploration steps ({len(explore_steps)}) — informational, not editable"):
        for s in explore_steps:
            st.write(f"- {s}")

    col_a, col_b, col_c = st.columns(3)
    approve = col_a.button("Approve & run", type="primary")
    skip_clean = col_b.button("Skip cleaning entirely")
    start_over = col_c.button("Start over")

    if start_over:
        _reset()
        st.rerun()

    if approve or skip_clean:
        new_clean_descriptions = [] if skip_clean else [ln.strip() for ln in edited_text.splitlines() if ln.strip()]
        state["plan"] = (
            [PlannedStep(stage="clean", description=d) for d in new_clean_descriptions]
            + [s for s in state["plan"] if s["stage"] != "clean"]
        )
        state["plan_approved"] = True

        on_progress = _make_progress_renderer()
        start = time.monotonic()
        with st.spinner("Executor -> Critic -> Synthesizer running..."):
            try:
                result = execute_analysis(checkpoint, on_progress=on_progress)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()
        st.session_state.elapsed_exec = time.monotonic() - start
        st.session_state.result = result
        st.rerun()

else:
    # ----- Stage 1: upload -> plan -----
    if st.button("Plan analysis", type="primary"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        chart_dir = tempfile.mkdtemp(prefix="autoanalyst_charts_")
        on_progress = _make_progress_renderer()

        start = time.monotonic()
        with st.spinner("Profiling dataset and planning..."):
            try:
                checkpoint = plan_analysis(
                    dataset_path=tmp_path,
                    dataset_name=uploaded.name,
                    model=model,
                    budget_usd=(budget if budget > 0 else None),
                    chart_dir=chart_dir,
                    on_progress=on_progress,
                )
            except Exception as e:
                st.error(f"Planning failed: {e}")
                st.stop()
        st.session_state.elapsed_plan = time.monotonic() - start
        st.session_state.checkpoint = checkpoint
        st.rerun()
