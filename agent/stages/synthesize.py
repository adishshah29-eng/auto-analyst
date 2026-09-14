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

Respond with a single ```json code block containing:
{"narrative": "<3-6 sentence plain-English summary, prose>",
 "non_obvious_findings": ["<finding 1 as a short standalone sentence>", "..."]}
"""

_USER_TEMPLATE = """Analysis summary for dataset "{dataset_name}":
{summary_json}
"""


def run(state: AnalysisState, tracker: CostTracker, model: str) -> None:
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
