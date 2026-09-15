"""Mocked regression tests for the 4-agent pipeline (Planner -> Executor ->
Critic -> Synthesizer) and the sandbox's infra-flake retry paths. No
network/API key required — every LLM call is a canned response, so these
test the orchestration logic, not model quality (see eval/run_eval.py for
that, which needs a real key)."""

from unittest.mock import patch

import pandas as pd
import pytest

from agent.llm import CostTracker, LLMResponse
from agent.sandbox import DEFAULT_TIMEOUT_SECONDS, SandboxResult
from agent.state import new_state

PLANNER_RESP = (
    "```json\n"
    '{"cleaning_steps": ["Impute missing age with median"], '
    '"exploration_steps": ["Check age distribution", "Correlate income and age"]}\n'
    "```"
)
CLEAN_RESP = (
    "```python\n"
    "df['age'] = df['age'].fillna(df['age'].median())\n"
    "cleaning_actions = ['Imputed missing age with median']\n"
    "```"
)
EXPLORE_RESP = (
    "```python\n"
    "findings = [\n"
    "  {'kind': 'distribution', 'description': 'Age is roughly normal around 35.', 'stats': {'mean': 35.0}},\n"
    "  {'kind': 'other', 'description': 'There are 200 rows.', 'stats': {}},\n"
    "]\n"
    "```"
)
CRITIC_FINDINGS_RESP = (
    '```json\n{"keep_indices": [0], "drop_reasons": {"1": "restates row count, not a real finding"}}\n```'
)
CHART_RESP = (
    "```python\n"
    "plt.figure()\nplt.hist(df['age'])\n"
    "chart_meta = [{'chart_type': 'histogram', 'question': 'What is the age distribution?'}]\n"
    "```"
)
SYNTH_RESP = (
    '```json\n{"narrative": "Age is normally distributed around 35.", "non_obvious_findings": ["Mean age is 35"]}\n```'
)
JUDGE_RESP = '```json\n{"grounded_score": 5, "non_obvious_score": 3, "actionable": true, "reasoning": "grounded"}\n```'


def _fake_call_llm(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2):
    if tracker is not None:
        tracker.add(0.0, 10, 5)
    if "Produce two lists" in system:
        return LLMResponse(text=PLANNER_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    if "cleans `df`" in user_message or "Implement EXACTLY" in user_message:
        return LLMResponse(text=CLEAN_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    if "Compute EXACTLY" in user_message or "explore `df`" in user_message:
        return LLMResponse(text=EXPLORE_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    if "Drop a finding if" in system:
        return LLMResponse(text=CRITIC_FINDINGS_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    if "create the matplotlib chart" in user_message:
        return LLMResponse(text=CHART_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    if "judge scoring" in system.lower():
        return LLMResponse(text=JUDGE_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)
    return LLMResponse(text=SYNTH_RESP, input_tokens=10, output_tokens=5, cost_usd=0.0)


@pytest.fixture
def mock_csv(tmp_path):
    df = pd.DataFrame({"age": [20, 30, None, 40] * 50, "income": [50000, 60000, 70000, 80000] * 50})
    path = tmp_path / "mock.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_full_pipeline_planner_executor_critic_synthesizer(mock_csv, tmp_path):
    with patch("agent.agents.planner.call_llm", side_effect=_fake_call_llm), \
         patch("agent.stages.common.call_llm", side_effect=_fake_call_llm), \
         patch("agent.agents.critic.call_llm", side_effect=_fake_call_llm), \
         patch("agent.stages.synthesize.call_llm", side_effect=_fake_call_llm):
        from agent.loop import run_analysis

        result = run_analysis(dataset_path=mock_csv, chart_dir=str(tmp_path / "charts"))

    state = result.state
    assert len(state["plan"]) == 3
    assert len([s for s in state["plan"] if s["stage"] == "clean"]) == 1
    assert len([s for s in state["plan"] if s["stage"] == "explore"]) == 2

    # the Critic must actually filter — this is a real gate, not a logged opinion
    assert len(state["findings"]) == 1, "critic should have dropped the trivial row-count finding"
    assert state["findings"][0]["description"].startswith("Age is roughly normal")
    assert state["critic_review"]["kept"] == 1
    assert state["critic_review"]["dropped"] == 1

    assert len(state["charts_generated"]) == 1
    assert state["narrative_review"]["grounded_score"] == 5


def test_timeout_retry_reuses_code_without_an_extra_llm_call():
    """A sandbox timeout is an infra stall, not a code bug — the retry
    should re-run the SAME code with more time, not spend an LLM call
    asking for a rewrite (see README "Key Learnings")."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    state = new_state("demo")
    tracker = CostTracker()

    call_count = {"n": 0}

    def fake_call_llm(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2):
        call_count["n"] += 1
        return LLMResponse(text="```python\nresult = 1 + 1\n```", input_tokens=10, output_tokens=5, cost_usd=0.0)

    sandbox_calls = []

    def fake_run_sandboxed(code, df, extra_context=None, capture_vars=None, **kwargs):
        sandbox_calls.append(kwargs.get("timeout"))
        if len(sandbox_calls) == 1:
            return SandboxResult(success=False, error="TimeoutError: execution exceeded 15s and was terminated.")
        return SandboxResult(success=True, output_vars={"result": 2})

    with patch("agent.stages.common.call_llm", side_effect=fake_call_llm), \
         patch("agent.stages.common.run_sandboxed", side_effect=fake_run_sandboxed):
        from agent.stages.common import SANDBOX_SYSTEM_PREAMBLE, _run_with_retry

        result, code = _run_with_retry(
            system=SANDBOX_SYSTEM_PREAMBLE, user_prompt="do it", df=df,
            capture_vars=["result"], stage="test", state=state, tracker=tracker, model="x",
        )

    assert call_count["n"] == 1, "LLM should be called once — the retry must not ask for a rewrite"
    assert len(sandbox_calls) == 2
    assert sandbox_calls[0] is None, "first attempt uses run_sandboxed's own default (no explicit timeout kwarg)"
    assert sandbox_calls[1] == DEFAULT_TIMEOUT_SECONDS * 2, "retry must double the timeout, not reuse the default"
    assert result.success


def test_critic_review_findings_filters_ungrounded_and_trivial_findings():
    from agent.agents import critic

    state = new_state("test")
    state["findings"] = [
        {"kind": "other", "description": "There are 500 rows.", "stats": {}},
        {"kind": "correlation", "description": "Strong correlation of 0.02 between A and B.", "stats": {"correlation": 0.02}},
        {"kind": "outlier", "description": "There are 15 outliers above 500.", "stats": {"n": 15}},
    ]
    fake_resp = LLMResponse(
        text='```json\n{"keep_indices": [2], "drop_reasons": {"0": "restates row count", "1": "contradicts its own stats"}}\n```',
        input_tokens=50, output_tokens=30, cost_usd=0.0,
    )
    with patch("agent.agents.critic.call_llm", return_value=fake_resp):
        review = critic.review_findings(state, tracker=None, model="x")

    assert review["kept"] == 1 and review["dropped"] == 2
    assert len(state["findings"]) == 1
    assert state["findings"][0]["description"].startswith("There are 15")


def test_critic_fails_open_on_unparseable_response():
    """A critic that can't parse its own output must never silently empty
    the findings list — fail open (keep everything), not closed."""
    from agent.agents import critic

    state = new_state("test2")
    state["findings"] = [{"kind": "other", "description": "x", "stats": {}}]
    bad_resp = LLMResponse(text="not json at all", input_tokens=10, output_tokens=5, cost_usd=0.0)
    with patch("agent.agents.critic.call_llm", return_value=bad_resp):
        critic.review_findings(state, tracker=None, model="x")

    assert len(state["findings"]) == 1
