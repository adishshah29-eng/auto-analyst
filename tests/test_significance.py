"""Pure-function tests for the deterministic significance gate — no LLM
mocking needed, this is arithmetic (see agent/agents/significance.py for
why it's split from the Critic's LLM-based review_findings)."""

from agent.agents.significance import flag_low_confidence_findings


def test_flags_small_subgroup_groupby_finding():
    """Regression test for the exact case observed live: a churn dataset's
    revenue/churn finding computed from ~60 customers, no fabrication, no
    mechanism to flag the sample as too small — until now."""
    findings = [
        {"kind": "groupby", "description": "High-revenue customers churn less.", "stats": {"n": 60, "rate": 0.033}, "caveat": ""},
    ]
    assert flag_low_confidence_findings(findings, min_n=30)[0]["caveat"] == ""  # 60 >= 30, not flagged
    flagged = flag_low_confidence_findings(findings, min_n=100)[0]
    assert flagged["caveat"] != ""  # 60 < 100, flagged
    assert "60" in flagged["caveat"]


def test_flags_finding_below_default_threshold():
    findings = [
        {"kind": "groupby", "description": "Enterprise plan users churn less.", "stats": {"n": 18}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings)
    assert "18" in out[0]["caveat"]
    assert "exploratory" in out[0]["caveat"]


def test_flags_weak_correlation():
    findings = [
        {"kind": "correlation", "description": "Age correlates with fare.", "stats": {"correlation": 0.02}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings)
    assert out[0]["caveat"] != ""
    assert "weak" in out[0]["caveat"]


def test_does_not_flag_strong_correlation_or_large_subgroup():
    findings = [
        {"kind": "correlation", "description": "NPS correlates with churn.", "stats": {"correlation": -0.265, "n": 1200}, "caveat": ""},
        {"kind": "groupby", "description": "North region has more sales.", "stats": {"n": 500}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings)
    assert all(f["caveat"] == "" for f in out)


def test_passes_through_findings_missing_the_relevant_stat():
    """A finding whose stats don't report n/correlation can't be gated on a
    number the code never produced — must pass through unflagged, not
    penalized for an absent field."""
    findings = [
        {"kind": "groupby", "description": "Some breakdown.", "stats": {"mean": 42.0}, "caveat": ""},
        {"kind": "distribution", "description": "Age is normal.", "stats": {"mean": 35.0}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings)
    assert all(f["caveat"] == "" for f in out)


def test_never_overwrites_an_existing_caveat():
    findings = [
        {"kind": "groupby", "description": "x", "stats": {"n": 5}, "caveat": "already flagged by something else"},
    ]
    out = flag_low_confidence_findings(findings)
    assert out[0]["caveat"] == "already flagged by something else"


def test_ungated_kinds_are_never_flagged():
    findings = [
        {"kind": "other", "description": "There are 500 rows.", "stats": {"n": 2}, "caveat": ""},
        {"kind": "distribution", "description": "x", "stats": {"n": 1}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings)
    assert all(f["caveat"] == "" for f in out)
