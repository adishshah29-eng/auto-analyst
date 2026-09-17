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
# For a rate/proportion claim, the standard normal-approximation rule of
# thumb is ~30 events, not ~30 rows — see _rate_event_count() for why that
# distinction is the whole ballgame here.
DEFAULT_MIN_EVENTS = 30

_GATED_KINDS = {"groupby", "correlation", "outlier"}

# Stats keys whose value is a rate/proportion rather than a raw quantity.
_RATE_KEY_HINTS = ("rate", "pct", "percent", "proportion", "share", "ratio")


def flag_low_confidence_findings(
    findings: list[Finding],
    min_n: int = DEFAULT_MIN_N,
    min_abs_correlation: float = DEFAULT_MIN_ABS_CORRELATION,
    n_rows: int | None = None,
) -> list[Finding]:
    """Returns a new list with `caveat` set on findings whose own stats show
    they're likely noise.

    `n_rows` is the dataset's total row count (from dataset_schema). It is
    what makes this gate trustworthy rather than decorative: see
    _subgroup_n_is_suspect() — a groupby/outlier finding reporting
    n == n_rows is reporting the dataset size, not the subgroup size, and
    must not be treated as a large, safe sample.

    A finding whose stats don't report the number this check needs is
    caveated as *unverifiable* for the gated kinds where the subgroup size
    is the whole point (groupby, outlier), and passed through for the rest.
    An absent sample size is a reason to trust a subgroup claim less, not a
    reason to wave it through."""
    flagged: list[Finding] = []
    for f in findings:
        caveat = _caveat_for(f, min_n, min_abs_correlation, n_rows)
        if caveat and not f.get("caveat"):
            f = {**f, "caveat": caveat}
        flagged.append(f)
    return flagged


def _subgroup_n_is_suspect(kind: str, n: float, n_rows: int | None) -> bool:
    """True when a finding about a SUBSET of the data reports the size of
    the WHOLE dataset as its n.

    This is the failure this gate was rebuilt around, caught on a live run
    rather than in a unit test. Asked to "include n, the number of rows the
    finding is based on", gemini-flash-lite reported n=1200 (the full row
    count) for every finding — including "LatAm exhibits the highest churn
    rate at 8.25%", which rests on 97 customers of whom 8 churned, and
    "Partner channel at 8.00%", which rests on 150 customers of whom 12
    churned. Both are sampling noise against a 5.83% base rate, and both
    are categories with no causal link to churn in the generator that made
    that dataset. The gate read n=1200 >= 30, cleared them, and they became
    headline claims in the narrative.

    A groupby finding about one category, or an outlier finding about the
    outliers, is by construction about fewer rows than the dataset. When it
    claims otherwise, the number is the model's mistake, not a large
    sample. Correlations are exempt: a pairwise-complete correlation over
    every row legitimately has n == n_rows."""
    if kind not in {"groupby", "outlier"} or n_rows is None:
        return False
    return n >= n_rows


def _rate_event_count(stats: dict, n: float) -> tuple[float, float] | None:
    """For a groupby finding stating a rate ("LatAm churns at 8.25%", n=97),
    returns (events, rate_as_percent) — here (8.0, 8.25). None if no rate.

    Why this exists, and why row count alone wasn't enough: the first
    version of this gate checked `n >= 30` and cleared LatAm's 8.25% churn
    rate because n=97. But 8.25% of 97 is *8 churned customers*, against a
    5.83% base rate across the dataset — roughly one standard error away,
    i.e. indistinguishable from noise. (Confirmed against the generator
    that made that dataset: region has no effect on churn at all. Same for
    the "Partner channel churns at 8.00%" claim — 12 events out of 150.)
    A rate's stability is governed by how many events it rests on, not how
    many rows were scanned, and n=97 hides that 8 completely. Checking rows
    instead of events is why the gate cleared every finding it existed to
    catch on its first live run."""
    for key, value in stats.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not any(hint in str(key).lower() for hint in _RATE_KEY_HINTS):
            continue
        # A rate arrives either as a fraction (0.0825) or as a percent
        # (8.25) — the key name says which, and anything above 1 can only
        # be a percent.
        is_percent = value > 1 or any(h in str(key).lower() for h in ("pct", "percent"))
        fraction = value / 100.0 if is_percent else float(value)
        if not 0.0 <= fraction <= 1.0:
            continue
        return fraction * n, fraction * 100.0
    return None


def _caveat_for(f: Finding, min_n: int, min_abs_correlation: float, n_rows: int | None) -> str:
    kind = f.get("kind", "")
    if kind not in _GATED_KINDS:
        return ""
    stats = f.get("stats") or {}
    if not isinstance(stats, dict):
        return ""

    n = stats.get("n")
    has_n = isinstance(n, (int, float)) and not isinstance(n, bool)

    if has_n and _subgroup_n_is_suspect(kind, float(n), n_rows):
        return (
            f"subgroup size not reported — the stated n ({int(n)}) is the full dataset row count, "
            "so this may rest on far fewer rows than it appears; treat as exploratory"
        )

    if has_n and n < min_n:
        return f"based on only {int(n)} rows (below the {min_n}-row confidence threshold) — treat as exploratory, not conclusive"

    if has_n and kind == "groupby":
        rate = _rate_event_count(stats, float(n))
        if rate is not None and rate[0] < DEFAULT_MIN_EVENTS:
            events, pct = rate
            return (
                f"the {pct:.2f}% figure rests on only about {int(round(events))} actual cases out of "
                f"{int(n)} rows — too few for the rate to be stable, so a gap this size is within "
                "sampling noise; treat as exploratory, not a real difference"
            )

    if not has_n and kind in {"groupby", "outlier"}:
        return "subgroup size not reported, so the sample behind this can't be checked — treat as exploratory"

    if kind == "correlation":
        corr = stats.get("correlation")
        if isinstance(corr, (int, float)) and abs(corr) < min_abs_correlation:
            return f"correlation is weak ({corr:.3f}), likely noise rather than a real relationship"

    return ""
