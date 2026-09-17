"""Deterministic (non-LLM) significance gate, run before the Critic's LLM
call in review_findings().

Why this is separate from the Critic: the Critic's LLM pass checks whether
a finding is *fabricated* — does the stated number actually appear in the
stats it was given. That's a different question from whether the number is
*trustworthy* — a correctly-computed statistic from a 60-row subgroup with
no causal basis can still score "grounded 5/5" under that check, because
nothing about "grounded" implies "significant". Caught live: a churn
dataset run produced "high-revenue customers above the 95th percentile
churn less (0.033)" from ~60 customers, computed correctly, no fabrication
— and no mechanism anywhere in the pipeline flagged that the sample was too
small to support the claim.

Sample-size and effect-size thresholds are arithmetic, not judgment calls —
paying an LLM call to compare an integer against a constant would be both
slower and less reliable than just doing it. This function never drops a
finding (dropping is the LLM Critic's call, informed by the caveat this
sets); it only annotates.
"""

from __future__ import annotations

from agent.state import Finding

DEFAULT_MIN_N = 30
DEFAULT_MIN_ABS_CORRELATION = 0.1

_GATED_KINDS = {"groupby", "correlation", "outlier"}


def flag_low_confidence_findings(
    findings: list[Finding],
    min_n: int = DEFAULT_MIN_N,
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
) -> list[Finding]:
    """Returns a new list with `caveat` set on findings whose own stats show
    they're likely noise. A finding whose stats don't report the number
    this check needs ("n" for a subgroup/pairwise count, "correlation" for
    a correlation strength) is passed through unflagged — there's nothing
    to gate on when the code didn't report it, and refusing to flag beats
    guessing."""
    flagged: list[Finding] = []
    for f in findings:
        caveat = _caveat_for(f, min_n, min_abs_correlation)
        if caveat and not f.get("caveat"):
            f = {**f, "caveat": caveat}
        flagged.append(f)
    return flagged


def _caveat_for(f: Finding, min_n: int, min_abs_correlation: float) -> str:
    kind = f.get("kind", "")
    if kind not in _GATED_KINDS:
        return ""
    stats = f.get("stats") or {}
    if not isinstance(stats, dict):
        return ""

    n = stats.get("n")
    if isinstance(n, (int, float)) and n < min_n:
        return f"based on only {int(n)} rows (below the {min_n}-row confidence threshold) — treat as exploratory, not conclusive"

    if kind == "correlation":
        corr = stats.get("correlation")
        if isinstance(corr, (int, float)) and abs(corr) < min_abs_correlation:
            return f"correlation is weak ({corr:.3f}), likely noise rather than a real relationship"

    return ""
