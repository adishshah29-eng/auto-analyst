"""Runs the full pipeline against every dataset in eval/test_datasets and
reports the metrics from README "Evaluation Plan": code execution success
rate (first try and after self-correction), a chart-appropriateness proxy
score, and latency/cost per dataset.

Requires ANTHROPIC_API_KEY to be set (this drives the real agent, not a
mock) — see .env.example. Run with:  python eval/run_eval.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm import DEFAULT_MODEL  # noqa: E402
from agent.loop import run_analysis  # noqa: E402
from agent.state import AnalysisState  # noqa: E402

TEST_DATASET_DIR = os.path.join(os.path.dirname(__file__), "test_datasets")

# A chart type is "appropriate" for a finding kind if it appears in this set.
# Deliberately a coarse, defensible rubric (README section 13 calls for a
# human rubric pass too) rather than an NLP match between finding text and
# chart title — good enough to catch the failure mode this metric exists
# for: the same 2-3 chart types produced regardless of what was found.
APPROPRIATE_CHART_TYPES = {
    "distribution": {"histogram", "box"},
    "outlier": {"histogram", "box", "scatter"},
    "correlation": {"scatter"},
    "groupby": {"bar", "box"},
    "other": {"histogram", "box", "scatter", "bar", "line"},
}


def execution_success_rates(state: AnalysisState) -> dict:
    history = state["code_history"]
    first_attempts = [s for s in history if not s["retried"]]
    retried_attempts = [s for s in history if s["retried"]]

    first_try_rate = (
        sum(s["success"] for s in first_attempts) / len(first_attempts) if first_attempts else None
    )

    # Final outcome per stage invocation = the last recorded attempt for that stage.
    last_by_order: list[bool] = []
    seen_stage_positions: dict[str, int] = {}
    for s in history:
        seen_stage_positions[s["stage"]] = len(last_by_order)
        last_by_order.append(s["success"])
    # crude but correct for this pipeline: each stage appears in one contiguous
    # run of <=2 entries (1 first try + up to 1 retry), so the last entry per
    # stage name is that stage's final success/fail.
    final_by_stage: dict[str, bool] = {}
    for s in history:
        final_by_stage[s["stage"]] = s["success"]

    final_success_rate = (
        sum(final_by_stage.values()) / len(final_by_stage) if final_by_stage else None
    )

    return {
        "first_try_success_rate": first_try_rate,
        "final_success_rate_after_retry": final_success_rate,
        "n_retries_used": len(retried_attempts),
        "n_code_generation_steps": len(history),
    }


def chart_appropriateness_score(state: AnalysisState) -> dict:
    finding_kinds = [f["kind"] for f in state["findings"]]
    charts = state["charts_generated"]
    if not charts:
        return {"score": None, "n_charts": 0, "note": "no charts generated"}

    # Score each chart against the union of "appropriate" types across all
    # finding kinds present (since charts aren't 1:1 indexed to findings).
    allowed: set[str] = set()
    for k in finding_kinds or ["other"]:
        allowed |= APPROPRIATE_CHART_TYPES.get(k, APPROPRIATE_CHART_TYPES["other"])

    appropriate = sum(1 for c in charts if c["chart_type"] in allowed)
    distinct_types = len({c["chart_type"] for c in charts})

    return {
        "score": round(appropriate / len(charts), 2),
        "n_charts": len(charts),
        "distinct_chart_types": distinct_types,
        "note": "proxy rubric — see APPROPRIATE_CHART_TYPES; pair with a human pass per dataset",
    }


def run_one(path: str, model: str, budget: float | None) -> dict:
    name = os.path.basename(path)
    print(f"\n=== {name} ===")
    start = time.monotonic()
    result = run_analysis(dataset_path=path, dataset_name=name, model=model, budget_usd=budget)
    elapsed = time.monotonic() - start
    state = result.state

    report = {
        "dataset": name,
        "rows": state["dataset_schema"].get("n_rows"),
        "cols": state["dataset_schema"].get("n_cols"),
        "latency_s": round(elapsed, 2),
        "cost_usd": round(result.tracker.total_cost_usd, 5),
        "input_tokens": result.tracker.total_input_tokens,
        "output_tokens": result.tracker.total_output_tokens,
        "n_findings": len(state["findings"]),
        "n_charts": len(state["charts_generated"]),
        "n_cleaning_actions": len(state["cleaning_actions_taken"]),
        "execution": execution_success_rates(state),
        "chart_appropriateness": chart_appropriateness_score(state),
        "stopped_early": result.stopped_early,
    }
    print(json.dumps(report, indent=2))
    return report


def print_summary_table(reports: list[dict]) -> None:
    print("\n\n## Results\n")
    print("| Dataset | Rows x Cols | First-try success | After retry | Charts | Appropriateness | Latency | Cost |")
    print("|---|---|---|---|---|---|---|---|")
    for r in reports:
        exe = r["execution"]
        chart = r["chart_appropriateness"]
        first = f"{exe['first_try_success_rate']:.0%}" if exe["first_try_success_rate"] is not None else "n/a"
        final = f"{exe['final_success_rate_after_retry']:.0%}" if exe["final_success_rate_after_retry"] is not None else "n/a"
        score = f"{chart['score']:.0%}" if chart["score"] is not None else "n/a"
        print(
            f"| {r['dataset']} | {r['rows']}x{r['cols']} | {first} | {final} | "
            f"{r['n_charts']} | {score} | {r['latency_s']}s | ${r['cost_usd']} |"
        )

    n = len(reports)
    if n:
        avg_latency = sum(r["latency_s"] for r in reports) / n
        avg_cost = sum(r["cost_usd"] for r in reports) / n
        print(f"\nAvg latency: {avg_latency:.1f}s  |  Avg cost: ${avg_cost:.4f}  |  Datasets: {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=float, default=1.0, help="per-dataset budget ceiling, USD")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set — this eval drives the real agent against real "
            "datasets and needs a key. Copy .env.example to .env and set it.",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(TEST_DATASET_DIR, "*.csv")))
    if not paths:
        print(f"No datasets found in {TEST_DATASET_DIR}. Run eval/generate_datasets.py first.")
        sys.exit(1)

    reports = [run_one(p, args.model, args.budget) for p in paths]
    print_summary_table(reports)

    with open(args.out, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
