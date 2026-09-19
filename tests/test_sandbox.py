"""Isolation tests for agent/sandbox.py — run these before trusting any
LLM-generated code near the sandbox. No network/API key required."""

import os

import pandas as pd
import pytest

from agent.sandbox import run_sandboxed


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "z", "w"]})


def test_success_and_capture_vars(df):
    r = run_sandboxed("result = df['a'].sum()\nprint('hello')", df, capture_vars=["result"])
    assert r.success, r.error
    assert r.stdout.strip() == "hello"
    assert r.output_vars["result"] == 6.0


def test_copy_on_inject_never_mutates_caller_df(df):
    r = run_sandboxed("df['a'] = 999", df)
    assert r.success
    assert df["a"].tolist()[:3] == [1, 2, 3]
    assert pd.isna(df["a"].tolist()[3])


def test_error_captured_with_traceback(df):
    r = run_sandboxed("1 / 0", df)
    assert not r.success
    assert "ZeroDivisionError" in r.error


def test_timeout_enforced(df):
    r = run_sandboxed("while True: pass", df, timeout=3)
    assert not r.success
    assert "Timeout" in r.error


def test_import_blocked(df):
    r = run_sandboxed("import os\nos.listdir('/')", df)
    assert not r.success
    assert "ImportError" in r.error or "not defined" in r.error


def test_open_blocked(df):
    r = run_sandboxed("open('/etc/passwd').read()", df)
    assert not r.success
    assert "NameError" in r.error


def test_chart_saved_to_disk(df, tmp_path):
    """Charts are Plotly figures collected into a `charts` list convention
    (there's no plt.get_fignums()-style global registry to fall back on) —
    the worker reads that list back and saves each as a standalone
    interactive HTML file."""
    code = "charts = [go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])]"
    r = run_sandboxed(code, df, chart_dir=str(tmp_path), chart_prefix="t")
    assert r.success, r.error
    assert len(r.chart_paths) == 1
    assert os.path.exists(r.chart_paths[0])
    assert r.chart_paths[0].endswith(".html")
    with open(r.chart_paths[0]) as f:
        html = f.read()
    assert "plotly" in html.lower()


def test_chart_ignores_non_figure_entries_in_the_charts_list(df, tmp_path):
    """A `charts` list entry that isn't a real Plotly figure (e.g. the model
    appended the wrong thing) is skipped rather than crashing the whole
    stage — one bad entry shouldn't lose every other chart."""
    code = "charts = [go.Figure(data=[go.Scatter(x=[1], y=[1])]), 'not a figure', 42]"
    r = run_sandboxed(code, df, chart_dir=str(tmp_path), chart_prefix="t2")
    assert r.success, r.error
    assert len(r.chart_paths) == 1


def test_severe_memory_pressure_reports_a_real_error_not_silent_crash(df):
    """RLIMIT_AS this tight makes even the child's own error handler
    struggle (see README "Key Learnings") — the sandbox must still surface
    *something* diagnosable rather than dying with zero information."""
    r = run_sandboxed("x = 1", df, timeout=10, memory_limit_mb=300)
    assert not r.success
    assert r.error
