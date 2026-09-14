"""Stage 5: Insight Synthesis. A separate, final LLM call — no code
execution here. It reads the structured findings + chart metadata the
earlier stages produced and writes the narrative. Kept distinct from the
analysis stages on purpose: stages 1-4 *produce* findings, this one
*explains* them. Mixing the two weakens both."""

from __future__ import annotations

import json

from agent.llm import CostTracker, call_llm, extract_json
from agent.state import AnalysisState, summarize_for_prompt

_SYSTEM = """You are a data analyst writing the final summary of an automated analysis for a
non-technical reader. You are given structured findings and chart descriptions, not raw data.
Write plain English. Prioritize the most surprising or decision-relevant findings over restating
row/column counts. If a finding only restates something obvious (e.g. "the dataset has N rows"),
leave it out of the narrative or mention it only in passing.

Ground every claim in the findings/schema JSON you are given below — never state a specific
number, percentage, or statistic that does not appear in that JSON. If the JSON's "findings" list
is empty or very sparse, say plainly that the automated analysis did not produce enough findings
to summarize, rather than inventing plausible-sounding numbers to fill out the narrative.

Respond with a single ```json code block containing:
{"narrative": "<3-6 sentence plain-English summary, prose>",
 "non_obvious_findings": ["<finding 1 as a short standalone sentence>", "..."]}
"""

_USER_TEMPLATE = """Analysis summary for dataset "{dataset_name}":
{summary_json}
"""

_NO_FINDINGS_NARRATIVE = (
    "The automated analysis did not produce any findings or charts for this dataset — the "
    "cleaning, exploration, and charting stages all failed to run successfully (see the "
    "execution log below for what went wrong at each step). There's nothing here to "
    "synthesize into a summary yet; check the errors and retry."
)


def run(state: AnalysisState, tracker: CostTracker, model: str) -> None:
    if not state["findings"] and not state["charts_generated"]:
        # Don't even call the LLM here: with nothing to summarize, the risk
        # is a plausible-sounding narrative with fabricated numbers rather
        # than an honest "this failed" — observed live on a deployed run
        # where every analysis stage failed yet the model still produced a
        # detailed narrative citing specific statistics that didn't come
        # from anywhere (see README "Key Learnings"). A hardcoded, honest
        # message beats a prompt instruction the model might not follow.
        state["narrative_summary"] = _NO_FINDINGS_NARRATIVE
        return

    summary = summarize_for_prompt(state)
    user_message = _USER_TEMPLATE.format(
        dataset_name=state["dataset_name"],
        summary_json=json.dumps(summary, default=str)[:8000],
    )

    resp = call_llm(system=_SYSTEM, user_message=user_message, tracker=tracker, model=model, max_tokens=1024)

    try:
        parsed = extract_json(resp.text)
        narrative = str(parsed.get("narrative", "")).strip()
        non_obvious = parsed.get("non_obvious_findings", [])
        if isinstance(non_obvious, list) and non_obvious:
            bullets = "\n".join(f"- {b}" for b in non_obvious)
            narrative = f"{narrative}\n\nKey non-obvious findings:\n{bullets}"
    except (json.JSONDecodeError, ValueError):
        narrative = resp.text.strip()

    state["narrative_summary"] = narrative
