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
    "charts = [go.Figure(data=[go.Histogram(x=df['age'])])]\n"
    "chart_meta = [{'chart_type': 'histogram', 'question': 'What is the age distribution?'}]\n"
    "```"
)
SYNTH_RESP = (
    '```json\n{"narrative": "Age is normally distributed around 35.", "non_obvious_findings": ["Mean age is 35"]}\n```'
)
JUDGE_RESP = '```json\n{"grounded_score": 5, "non_obvious_score": 3, "actionable": true, "reasoning": "grounded"}\n```'


def _fake_call_llm(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2, stage=""):
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
    if "create the Plotly chart" in user_message:
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

    def fake_call_llm(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2, stage=""):
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
    """The LLM pass still fully controls findings the significance gate has
    no opinion on: a trivial restatement and a distribution claim that
    misstates its own stats (neither "groupby"/"correlation"/"outlier", so
    neither can ever carry a caveat) are dropped exactly as the model says."""
    from agent.agents import critic

    state = new_state("test")
    state["findings"] = [
        {"kind": "other", "description": "There are 500 rows.", "stats": {}},
        {"kind": "distribution", "description": "Age is bimodal with peaks at 20 and 80.", "stats": {"mean": 35.0, "std": 5.0}},
        {"kind": "groupby", "description": "North region has 500 customers.", "stats": {"n": 500}},
    ]
    fake_resp = LLMResponse(
        text='```json\n{"keep_indices": [2], "drop_reasons": {"0": "restates row count", "1": "std of 5 around a mean of 35 doesn\'t support a bimodal claim"}}\n```',
        input_tokens=50, output_tokens=30, cost_usd=0.0,
    )
    with patch("agent.agents.critic.call_llm", return_value=fake_resp):
        review = critic.review_findings(state, tracker=None, model="x")

    assert review["kept"] == 1 and review["dropped"] == 2
    assert len(state["findings"]) == 1
    assert state["findings"][0]["description"].startswith("North region")


def test_critic_never_drops_a_caveated_finding_even_if_the_llm_tries_to():
    """Policy decision, enforced in code rather than left to the LLM's
    compliance with an instruction: a finding the deterministic
    significance gate has already caveated is NEVER dropped by the Critic's
    LLM pass, regardless of what the LLM decides. Before this was enforced,
    the exact same prompt ("weigh the caveat, don't treat it as automatic
    grounds to drop") produced two different outcomes on two live runs —
    one dropped a small-sample finding outright, another kept and hedged an
    equivalent one. That inconsistency, not the underlying arithmetic, was
    the bug — see README "Critic & LLM-as-Judge".

    This mock LLM tries to drop ALL THREE findings, including the caveated
    one, citing the caveat's own reasoning ("small sample") as its excuse —
    exactly the failure mode this test guards against."""
    from agent.agents import critic

    state = new_state("test")
    state["dataset_schema"] = {"n_rows": 1200}
    state["findings"] = [
        {"kind": "other", "description": "There are 1200 rows.", "stats": {}},
        {"kind": "distribution", "description": "Nonsense claim.", "stats": {"mean": 1.0}},
        {"kind": "outlier", "description": "There are 15 outliers.", "stats": {"n": 15}},
    ]
    fake_resp = LLMResponse(
        text='```json\n{"keep_indices": [], "drop_reasons": {"0": "restates row count", "1": "unsupported", "2": "small sample, weak evidence"}}\n```',
        input_tokens=50, output_tokens=30, cost_usd=0.0,
    )
    with patch("agent.agents.critic.call_llm", return_value=fake_resp):
        review = critic.review_findings(state, tracker=None, model="x")

    assert review["kept"] == 1, "the caveated finding must survive even though the LLM tried to drop it"
    assert review["dropped"] == 2
    assert len(state["findings"]) == 1
    assert state["findings"][0]["description"] == "There are 15 outliers."
    assert state["findings"][0]["caveat"] != ""


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


def test_user_goal_reaches_planner_explore_chart_and_synthesize_prompts():
    """The intent gate only means anything if the goal actually steers the
    downstream stages. Asserts the goal text reaches all four prompts that
    are supposed to act on it — a goal stored in state but never threaded
    into a prompt would look fine in the UI and change nothing about the
    output."""
    from agent.agents import planner
    from agent.stages import chart, explore, synthesize

    goal = "WHICH-REP-IS-BEST-SENTINEL"
    prompts: dict[str, str] = {}

    def capture(key):
        def fake(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2, stage=""):
            prompts[key] = user_message
            # planner/synthesize parse JSON; explore/chart parse a code block
            if key in ("planner", "synthesize"):
                return LLMResponse(text='```json\n{}\n```', input_tokens=1, output_tokens=1, cost_usd=0.0)
            return LLMResponse(text="```python\nfindings = []\n```", input_tokens=1, output_tokens=1, cost_usd=0.0)
        return fake

    state = new_state("goal_test")
    state["dataset_schema"] = {"n_rows": 10, "columns": {}}
    state["user_goal"] = goal
    state["findings"] = [{"kind": "groupby", "description": "d", "stats": {}}]
    state["narrative_summary"] = "n"
    df = pd.DataFrame({"a": [1, 2, 3]})

    with patch("agent.agents.planner.call_llm", side_effect=capture("planner")):
        planner.plan(state, tracker=None, model="x")
    with patch("agent.stages.common.call_llm", side_effect=capture("explore")), \
         patch("agent.stages.common.run_sandboxed", return_value=SandboxResult(success=True, output_vars={})):
        explore.run(state, df, tracker=None, model="x", planned_steps=["step one"])
    with patch("agent.stages.common.call_llm", side_effect=capture("chart")), \
         patch("agent.stages.common.run_sandboxed", return_value=SandboxResult(success=True, output_vars={})):
        chart.run(state, df, tracker=None, model="x", chart_dir="/tmp")
    with patch("agent.stages.synthesize.call_llm", side_effect=capture("synthesize")):
        synthesize.run(state, tracker=None, model="x")

    for stage in ("planner", "explore", "chart", "synthesize"):
        assert goal in prompts[stage], f"user_goal never reached the {stage} prompt"


def test_review_narrative_shows_the_judge_the_schema_too():
    """Regression test: review_narrative() originally only showed the judge
    `findings` + `cleaning_actions_taken`, so a narrative correctly citing a
    schema-level fact (row/column counts, a date range, a category
    distribution — all legitimately available to the Synthesizer via
    summarize_for_prompt) got scored as fabrication. Caught live: a real
    narrative citing "94 unique roads... spanning Jan 1 to Mar 26" (schema
    facts) scored grounded_score 3/5 with reasoning calling them
    unsupported. The fix passes dataset_schema to the judge too — this test
    locks in that the schema actually reaches the prompt, not just that
    *some* JSON does."""
    from agent.agents import critic

    state = new_state("test3")
    state["dataset_schema"] = {"n_rows": 9000, "columns": {"road": {"n_unique": 94}}}
    state["findings"] = []
    state["cleaning_actions_taken"] = []
    state["narrative_summary"] = "The dataset spans 94 unique roads."

    captured = {}

    def fake_call_llm(system, user_message, tracker=None, model=None, max_tokens=2048, temperature=0.2, stage=""):
        captured["user_message"] = user_message
        return LLMResponse(
            text='```json\n{"grounded_score": 5, "non_obvious_score": 2, "actionable": false, "reasoning": "ok"}\n```',
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )

    with patch("agent.agents.critic.call_llm", side_effect=fake_call_llm):
        score = critic.review_narrative(state, tracker=None, model="x")

    assert "94" in captured["user_message"], "the schema fact the narrative cites must reach the judge's prompt"
    assert score["grounded_score"] == 5
