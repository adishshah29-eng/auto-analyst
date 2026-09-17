"""Structured run tracing: one append-only JSONL file per analysis run,
written to outputs/runs/<run_id>.jsonl (already gitignored — this is what
that directory was reserved for).

Why this exists: every bug found while building this pipeline (the sandbox
queue deadlock, the judge-scoping bug, a silently dropped chart_dir, a
dotenv-ordering bug) was diagnosed by re-reading code or a live run — there
was no persisted record of what a run actually did, so "it gave a weird
answer yesterday" was unreproducible. RunTracer makes each run replayable
after the fact: every LLM call (prompts, response, tokens, cost) and every
sandbox execution (code, success, error) for a given run_id, in order.

Deliberately NOT a hosted tracing service (Langfuse/OpenTelemetry) — that
needs a hosted account this project doesn't have credentials for. The
per-event-type method shape below (log_llm_call / log_sandbox_run /
log_stage_boundary) is the swap-in seam: replace the body of _write() with
an exporter call and every call site is already instrumented.

A tracing failure must never break an analysis run — every write is
best-effort and swallows I/O errors.
"""

from __future__ import annotations

import json
import os
import time
import uuid

TRACE_DIR = os.environ.get("TRACE_DIR", os.path.join("outputs", "runs"))

# Consistent with the existing style throughout agent/ of truncating
# JSON-serialized payloads with a fixed slice (e.g. profile_json[:6000] in
# explore.py) rather than a separate config knob per field.
_PREVIEW_CHARS = 1500


class RunTracer:
    def __init__(self, dataset_name: str, run_id: str | None = None, enabled: bool = True):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.enabled = enabled
        self.path = os.path.join(TRACE_DIR, f"{self.run_id}.jsonl")
        self._t0 = time.monotonic()
        if self.enabled:
            try:
                os.makedirs(TRACE_DIR, exist_ok=True)
            except OSError:
                self.enabled = False
        self._write({"event": "run_start", "dataset": dataset_name})

    def _write(self, payload: dict) -> None:
        if not self.enabled:
            return
        record = {
            "run_id": self.run_id,
            "ts": time.time(),
            "elapsed_s": round(time.monotonic() - self._t0, 3),
            **payload,
        }
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            # Tracing is a diagnostic aid, not a pipeline dependency — a
            # read-only filesystem or a full disk must not fail the run.
            self.enabled = False

    def log_llm_call(
        self,
        stage: str,
        model: str,
        system: str,
        user_message: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        self._write(
            {
                "event": "llm_call",
                "stage": stage,
                "model": model,
                "system_preview": system[:_PREVIEW_CHARS],
                "user_preview": user_message[:_PREVIEW_CHARS],
                "response_preview": response_text[:_PREVIEW_CHARS],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
            }
        )

    def log_sandbox_run(self, stage: str, code: str, success: bool, error: str | None) -> None:
        self._write(
            {
                "event": "sandbox_run",
                "stage": stage,
                "code": code[:_PREVIEW_CHARS * 2],
                "success": success,
                "error": (error or "")[:_PREVIEW_CHARS],
            }
        )

    def log_stage_boundary(self, stage: str, event: str, message: str = "") -> None:
        # event: "start" | "end"
        self._write({"event": f"stage_{event}", "stage": stage, "message": message})
