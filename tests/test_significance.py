"""Pure-function tests for the deterministic significance gate — no LLM
mocking needed, this is arithmetic (see agent/agents/significance.py for
why it's split from the Critic's LLM-based review_findings)."""

from agent.agents.significance import flag_low_confidence_findings


def test_flags_the_original_revenue_churn_finding_that_motivated_this_gate():
    """The finding this whole gate was built for: a live churn run produced
    "high-revenue customers above the 95th percentile churn less (0.033)",
    computed correctly, scored grounded 5/5 by the judge, and resting on
    ~60 customers.

    The first version of this test asserted that n=60 should NOT be flagged
    at a 30-row threshold — encoding the row-count assumption the gate was
    then built on. That assumption was wrong, and the event-count check
    shows why in one number: 3.3% of 60 customers is *2 churned customers*.
    A two-customer difference was being reported as a business insight.
    Row count said "60, comfortably above 30"; event count says "2"."""
    findings = [
        {"kind": "groupby", "description": "High-revenue customers churn less.", "stats": {"n": 60, "rate": 0.033}, "caveat": ""},
    ]
    caveat = flag_low_confidence_findings(findings, min_n=30)[0]["caveat"]
    assert caveat != "", "a rate resting on 2 cases must be flagged regardless of the row count"
    assert "2 actual cases" in caveat


def test_flags_finding_below_the_row_count_threshold_too():
    """The row-count check still earns its place for subgroup findings that
    state no rate at all — it just can't be the only check."""
    findings = [
        {"kind": "groupby", "description": "Enterprise users churn less.", "stats": {"n": 18}, "caveat": ""},
    ]
    flagged = flag_low_confidence_findings(findings, min_n=30)[0]
    assert "18" in flagged["caveat"]


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


def test_subgroup_finding_reporting_the_whole_dataset_size_is_flagged():
    """The failure this gate was rebuilt around, caught on a live run rather
    than in a unit test: asked to report "n, the number of rows the finding
    is based on", gemini-flash-lite reported n=1200 (the full row count)
    for EVERY finding — including "LatAm exhibits the highest churn rate at
    8.25%", which actually rests on 97 customers of whom 8 churned. The
    gate read 1200 >= 30, cleared it, and it became a headline claim in the
    narrative. A groupby claim about one category is by construction about
    fewer rows than the dataset; claiming otherwise is the model's mistake,
    not a large sample."""
    findings = [
        {"kind": "groupby", "description": "LatAm has the highest churn rate at 8.25%.", "stats": {"n": 1200, "churn_rate": 0.0825}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] != "", "a subgroup claim reporting the full row count must not be cleared"
    assert "full dataset row count" in out[0]["caveat"]


def test_correlation_over_every_row_may_legitimately_equal_n_rows():
    """Unlike a subgroup, a pairwise-complete correlation across the whole
    frame genuinely has n == n_rows — must NOT be flagged, or the gate
    would caveat every correlation in every dataset."""
    findings = [
        {"kind": "correlation", "description": "NPS correlates with churn.", "stats": {"correlation": -0.265, "n": 1200}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] == ""


def test_subgroup_finding_with_a_real_subgroup_size_is_not_flagged():
    findings = [
        {"kind": "groupby", "description": "North America churn is 5.95%.", "stats": {"n": 504}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] == ""


def test_rate_claim_is_gated_on_event_count_not_row_count():
    """Second live failure, after the prompt fix made the model report real
    subgroup sizes: n=97 clears a 30-row threshold, but "LatAm churns at
    8.25%" with n=97 is *8 churned customers* against a 5.83% base rate —
    about one standard error, i.e. noise, and region has no effect on churn
    in the generator that produced that dataset. Row count was the wrong
    statistic for a rate claim; the event count behind it is the right one."""
    findings = [
        {"kind": "groupby", "description": "LatAm churns at 8.25%.", "stats": {"churn_rate_pct": 8.25, "n": 97}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] != "", "a rate resting on 8 events must not be cleared just because n=97"
    assert "8 actual cases" in out[0]["caveat"]


def test_rate_claim_with_enough_events_passes():
    """31 events out of 473 — a stable enough rate to state plainly. The
    gate has to leave these alone or every caveat becomes noise."""
    findings = [
        {"kind": "groupby", "description": "Organic churns at 6.55%.", "stats": {"churn_rate_pct": 6.55, "n": 473}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] == ""


def test_rate_expressed_as_a_fraction_is_handled_like_a_percentage():
    """Models report a rate either way (0.0825 or 8.25) — both must resolve
    to the same 8 events, or the check silently misses one of the forms."""
    as_fraction = [{"kind": "groupby", "description": "x", "stats": {"churn_rate": 0.0825, "n": 97}, "caveat": ""}]
    as_percent = [{"kind": "groupby", "description": "x", "stats": {"churn_rate_pct": 8.25, "n": 97}, "caveat": ""}]
    assert "8 actual cases" in flag_low_confidence_findings(as_fraction, n_rows=1200)[0]["caveat"]
    assert "8 actual cases" in flag_low_confidence_findings(as_percent, n_rows=1200)[0]["caveat"]


def test_groupby_of_means_without_a_rate_is_not_event_gated():
    """A mean-comparison groupby has no events to count — it must fall
    through to the other checks rather than being mis-parsed as a rate."""
    findings = [
        {"kind": "groupby", "description": "Churned avg tickets 2.73 vs 1.76.", "stats": {"tickets_churned": 2.73, "tickets_active": 1.76, "n": 400}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert out[0]["caveat"] == ""


def test_subgroup_finding_with_no_n_at_all_is_flagged_as_unverifiable():
    """An absent sample size is a reason to trust a subgroup claim less, not
    a reason to wave it through — the gate fails safe, since silently
    clearing an unverifiable claim is the exact behavior that made the
    first version of this decorative."""
    findings = [
        {"kind": "groupby", "description": "Some breakdown.", "stats": {"mean": 42.0}, "caveat": ""},
        {"kind": "outlier", "description": "Revenue has outliers.", "stats": {"max": 2934.8}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
    assert all(f["caveat"] != "" for f in out)
    assert "can't be checked" in out[0]["caveat"]


def test_non_subgroup_kinds_without_n_still_pass_through():
    findings = [
        {"kind": "distribution", "description": "Age is normal.", "stats": {"mean": 35.0}, "caveat": ""},
        {"kind": "correlation", "description": "Strong link.", "stats": {"correlation": 0.65}, "caveat": ""},
    ]
    out = flag_low_confidence_findings(findings, n_rows=1200)
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
