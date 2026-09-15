"""Critic agent: the third of the four agents (Planner -> Executor ->
Critic -> Synthesizer). Two related jobs, both scoped to state — never
raw data — so this agent is as safe from prompt injection as every other
stage:

- `review_findings()` is a live, in-loop quality gate: it runs right after
  Explore and before Chart, so a finding it drops as trivial or wrong
  never reaches a chart or the final narrative. Mutates state["findings"]
  in place — this is a real filter, not just a logged opinion.
- `review_narrative()` is a reusable LLM-as-judge: scores the FINISHED
  narrative for whether it's grounded in the findings it was given and
  whether it says anything non-obvious. Used live (for transparency in the
  UI) and, unmodified, as eval/run_eval.py's "insight relevance" metric —
  same code, two call sites, so the eval number means the same thing the
  live badge means.
"""

from __future__ import annotations

import json

from agent.llm import CostTracker, call_llm, extract_json
from agent.state import AnalysisState, CriticReview

_REVIEW_FINDINGS_SYSTEM = """You are a critic agent reviewing a data analysis pipeline's output before
it reaches the final report. You are given a list of findings (aggregated stats only, no raw data)
and must decide which are worth keeping.

Drop a finding if it:
- only restates something obvious (e.g. "there are 500 rows") rather than something specific to
  the data's content
- is not actually supported by the stats attached to it (e.g. calls a correlation "strong" when
  the stats show it's near zero)
- duplicates another finding in substance

Keep a finding if it's specific, supported by its own stats, and says something non-generic about
this dataset.

Respond with a single ```json code block:
{"keep_indices": [0, 2, 3, ...], "drop_reasons": {"1": "restates row count, not a real finding", ...}}
Indices are 0-based positions in the findings list you were given, in order.
"""

_REVIEW_FINDINGS_USER = """Findings to review:
{findings_json}
"""

_REVIEW_NARRATIVE_SYSTEM = """You are a judge scoring a data-analysis agent's final narrative summary,
for an evaluation harness. You are given the structured findings the narrative was supposed to be
based on, and the narrative itself.

Score on three axes:
- "grounded_score" (1-5): does every specific number/claim in the narrative trace back to something
  in the findings? 5 = fully grounded, 1 = the narrative states specifics not present in the findings
  at all (fabrication).
- "non_obvious_score" (1-5): does the narrative surface something a reader wouldn't get from just
  glancing at row/column counts? 5 = genuinely surprising or decision-relevant, 1 = pure boilerplate
  ("the dataset has N rows and M columns").
- "actionable" (true/false): would this narrative plausibly change a real decision, or is it just
  descriptive filler?

Respond with a single ```json code block:
{"grounded_score": 1-5, "non_obvious_score": 1-5, "actionable": true/false, "reasoning": "<1-2 sentences>"}
"""

_REVIEW_NARRATIVE_USER = """The narrative below is allowed to draw on BOTH of these sources — ground
your score against both, not findings alone (a narrative correctly mentioning a cleaning action,
e.g. a dropped column, is grounded even though that fact lives here and not in the findings list):

Cleaning actions taken:
{cleaning_actions_json}

Findings from exploration:
{findings_json}

Narrative to score:
{narrative}
"""


def review_findings(state: AnalysisState, tracker: CostTracker, model: str) -> CriticReview:
    """Filters state["findings"] in place. Runs between Explore and Chart
    so a dropped finding never gets a chart or reaches synthesis."""
    findings = state["findings"]
    if not findings:
        review = CriticReview(kept=0, dropped=0, reasons=[])
        state["critic_review"] = review
        return review

    findings_json = json.dumps(findings, default=str)[:6000]
    resp = call_llm(
        system=_REVIEW_FINDINGS_SYSTEM,
        user_message=_REVIEW_FINDINGS_USER.format(findings_json=findings_json),
        tracker=tracker,
        model=model,
        max_tokens=1024,
    )

    try:
        parsed = extract_json(resp.text)
        keep_indices = {int(i) for i in parsed.get("keep_indices", [])}
        drop_reasons = parsed.get("drop_reasons", {}) or {}
    except (json.JSONDecodeError, ValueError, TypeError):
        # A critic that fails to parse shouldn't drop everything — keep
        # all findings rather than let a parsing bug silently empty the
        # report (fail open, not closed, for a quality gate like this).
        review = CriticReview(kept=len(findings), dropped=0, reasons=[])
        state["critic_review"] = review
        return review

    reasons = [str(v) for v in drop_reasons.values()]
    kept_findings = [f for i, f in enumerate(findings) if i in keep_indices]

    review = CriticReview(kept=len(kept_findings), dropped=len(findings) - len(kept_findings), reasons=reasons)
    state["findings"] = kept_findings if kept_findings else findings  # never drop to zero silently
    if not kept_findings:
        review["kept"] = len(findings)
        review["dropped"] = 0
    state["critic_review"] = review
    return review


def review_narrative(state: AnalysisState, tracker: CostTracker, model: str) -> dict:
    """LLM-as-judge over the finished narrative. Read-only — never mutates
    state. Reused verbatim by eval/run_eval.py so the eval's "insight
    relevance" number and the live UI's judge badge mean the same thing."""
    if not state["narrative_summary"]:
        return {"grounded_score": None, "non_obvious_score": None, "actionable": None, "reasoning": "no narrative to score"}

    findings_json = json.dumps(state["findings"], default=str)[:6000]
    cleaning_actions_json = json.dumps(state["cleaning_actions_taken"], default=str)[:2000]
    resp = call_llm(
        system=_REVIEW_NARRATIVE_SYSTEM,
        user_message=_REVIEW_NARRATIVE_USER.format(
            findings_json=findings_json,
            cleaning_actions_json=cleaning_actions_json,
            narrative=state["narrative_summary"][:4000],
        ),
        tracker=tracker,
        model=model,
        max_tokens=512,
    )

    try:
        parsed = extract_json(resp.text)
        return {
            "grounded_score": parsed.get("grounded_score"),
            "non_obvious_score": parsed.get("non_obvious_score"),
            "actionable": parsed.get("actionable"),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except (json.JSONDecodeError, ValueError):
        return {"grounded_score": None, "non_obvious_score": None, "actionable": None, "reasoning": "judge response unparseable"}
