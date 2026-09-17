"""Thin wrapper around the model API: one place for model calls,
token/cost accounting, and a budget ceiling.

Supports two providers behind one interface — Anthropic (Claude) and
Google (Gemini, including the free AI Studio tier). The provider is
inferred from the model name (a "gemini*" model routes to Google,
everything else to Anthropic) or set explicitly via LLM_PROVIDER.

Reasoning-prompt safety lives at the call site (stages pass in schema +
aggregated stats via agent.state.summarize_for_prompt), not here — this
module just makes the call and meters it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = os.environ.get("ANALYSIS_MODEL", "claude-sonnet-5")

# Approximate list pricing in USD per million tokens. These change over
# time and vary by model — treat as a configurable estimate for budgeting,
# not a billing source of truth. Override per-model via
# ANALYSIS_MODEL_PRICE_IN / ANALYSIS_MODEL_PRICE_OUT env vars.
# Verify current Anthropic numbers at https://claude.com/pricing.
_ANTHROPIC_PRICING_PER_MTOK = {
    "claude-opus-5": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-fable-5-1": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}

# Google's AI Studio free tier (a plain GOOGLE_API_KEY, not Vertex billing)
# costs $0 and is rate-limited rather than billed, so 0.0 is the accurate
# default here — not a placeholder like the Anthropic table above. If
# you're on paid Gemini API billing, set ANALYSIS_MODEL_PRICE_IN/_OUT to
# get real cost estimates instead of zeros.
_GOOGLE_PRICING_PER_MTOK: dict[str, tuple[float, float]] = {}

_DEFAULT_PRICE_FALLBACK = {"anthropic": (3.00, 15.00), "google": (0.0, 0.0)}


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Accumulates spend across a whole analysis run and enforces a ceiling.

    Optionally carries a RunTracer (agent.tracing) so call_llm() can log
    every LLM call against the same run_id this tracker is already threaded
    through — no new parameter needed at any of the ~10 call sites."""

    def __init__(self, budget_usd: float | None = None, tracer: Any | None = None):
        self.budget_usd = budget_usd
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tracer = tracer

    def add(self, cost_usd: float, input_tokens: int, output_tokens: int) -> None:
        self.total_cost_usd += cost_usd
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def check_budget(self) -> None:
        if self.budget_usd is not None and self.total_cost_usd >= self.budget_usd:
            raise BudgetExceededError(
                f"Analysis stopped: spent ${self.total_cost_usd:.4f}, "
                f"budget ceiling was ${self.budget_usd:.4f}."
            )


def infer_provider(model: str) -> str:
    override = os.environ.get("LLM_PROVIDER")
    if override:
        return override.strip().lower()
    return "google" if model.lower().startswith("gemini") else "anthropic"


def _price_for(model: str, provider: str) -> tuple[float, float]:
    env_in = os.environ.get("ANALYSIS_MODEL_PRICE_IN")
    env_out = os.environ.get("ANALYSIS_MODEL_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    table = _ANTHROPIC_PRICING_PER_MTOK if provider == "anthropic" else _GOOGLE_PRICING_PER_MTOK
    return table.get(model, _DEFAULT_PRICE_FALLBACK[provider])


_anthropic_client = None
_google_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set ANALYSIS_MODEL to a gemini-* model to use Google instead."
            )
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_google_client():
    global _google_client
    if _google_client is None:
        from google import genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your (free) "
                "AI Studio key: https://aistudio.google.com/apikey"
            )
        _google_client = genai.Client(api_key=api_key)
    return _google_client


def _call_anthropic(
    system: str, user_message: str, model: str, max_tokens: int, temperature: float
) -> tuple[str, int, int]:
    client = _get_anthropic_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _call_google(
    system: str, user_message: str, model: str, max_tokens: int, temperature: float
) -> tuple[str, int, int]:
    from google.genai import types

    client = _get_google_client()
    resp = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = resp.text or ""
    usage = resp.usage_metadata
    input_tokens = (usage.prompt_token_count or 0) if usage else 0
    output_tokens = (usage.candidates_token_count or 0) if usage else 0
    return text, input_tokens, output_tokens


def call_llm(
    system: str,
    user_message: str,
    tracker: CostTracker | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    stage: str = "",
) -> LLMResponse:
    """One model call, metered. Raises BudgetExceededError up front if the
    tracker already reports the ceiling as spent.

    `stage` is purely a tracing label (e.g. "plan", "explore", "synthesize")
    — it has no effect on the call itself. Passed through to tracker.tracer
    if one is attached, so a run's trace file can be read stage-by-stage."""
    if tracker is not None:
        tracker.check_budget()

    provider = infer_provider(model)
    if provider == "google":
        text, input_tokens, output_tokens = _call_google(system, user_message, model, max_tokens, temperature)
    elif provider == "anthropic":
        text, input_tokens, output_tokens = _call_anthropic(system, user_message, model, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (expected 'anthropic' or 'google')")

    price_in, price_out = _price_for(model, provider)
    cost = (input_tokens / 1_000_000) * price_in + (output_tokens / 1_000_000) * price_out

    if tracker is not None:
        tracker.add(cost, input_tokens, output_tokens)
        tracker.check_budget()
        if tracker.tracer is not None:
            tracker.tracer.log_llm_call(
                stage=stage,
                model=model,
                system=system,
                user_message=user_message,
                response_text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )

    return LLMResponse(text=text, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost)


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced ```python``` block out of an LLM response.
    Falls back to the raw text if the model didn't fence it."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def extract_json(text: str) -> dict:
    match = _JSON_BLOCK_RE.search(text)
    raw = match.group(1).strip() if match else text.strip()
    return json.loads(raw)
