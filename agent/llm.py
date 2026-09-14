"""Thin wrapper around the Anthropic API: one place for model calls,
token/cost accounting, and a budget ceiling.

Reasoning-prompt safety lives at the call site (stages pass in schema +
aggregated stats via agent.state.summarize_for_prompt), not here — this
module just makes the call and meters it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = os.environ.get("ANALYSIS_MODEL", "claude-sonnet-5")

# Approximate list pricing in USD per million tokens. These change over
# time and vary by model — treat as a configurable estimate for budgeting,
# not a billing source of truth. Verify current numbers at
# https://claude.com/pricing and update here (or override via
# ANALYSIS_MODEL_PRICE_IN / ANALYSIS_MODEL_PRICE_OUT env vars).
_DEFAULT_PRICING_PER_MTOK = {
    "claude-opus-5": (15.00, 75.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-fable-5-1": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Accumulates spend across a whole analysis run and enforces a ceiling."""

    def __init__(self, budget_usd: float | None = None):
        self.budget_usd = budget_usd
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

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


def _price_for(model: str) -> tuple[float, float]:
    env_in = os.environ.get("ANALYSIS_MODEL_PRICE_IN")
    env_out = os.environ.get("ANALYSIS_MODEL_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    return _DEFAULT_PRICING_PER_MTOK.get(model, (3.00, 15.00))


_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_llm(
    system: str,
    user_message: str,
    tracker: CostTracker | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.2,
) -> LLMResponse:
    """One Claude call, metered. Raises BudgetExceededError up front if the
    tracker already reports the ceiling as spent."""
    if tracker is not None:
        tracker.check_budget()

    client = _get_client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")

    price_in, price_out = _price_for(model)
    cost = (resp.usage.input_tokens / 1_000_000) * price_in + (
        resp.usage.output_tokens / 1_000_000
    ) * price_out

    if tracker is not None:
        tracker.add(cost, resp.usage.input_tokens, resp.usage.output_tokens)
        tracker.check_budget()

    return LLMResponse(
        text=text,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        cost_usd=cost,
    )


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
