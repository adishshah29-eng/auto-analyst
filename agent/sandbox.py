"""Restricted code-execution sandbox for agent-generated Python.

Design goals (see README "Sandbox & Security"):
  - The generated snippet only ever sees a confined namespace (pd, np, plt,
    a *copy* of the dataframe, and a small safe-builtins set) — never the
    real process globals, filesystem-wide imports, or the original df.
  - Execution happens in a separate process so a timeout can actually kill
    it, and so a memory blow-up in the snippet can't take down the agent.
  - Everything is captured: stdout, the requested output variables, and a
    full traceback on failure — nothing is swallowed.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue
import resource
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: never try to open a display
import matplotlib.pyplot as plt  # noqa: E402

# Overridable via env vars so a memory-constrained host (e.g. Streamlit
# Community Cloud's free tier, ~1GB total for the whole app) can lower
# these without a code change — the local/default values assume a normal
# dev machine.
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", 15))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", 1024))

# Deliberately small: enough for pandas/numpy/matplotlib code to run,
# not enough to import arbitrary modules, touch the filesystem outside
# outputs/, open sockets, or spawn processes.
_SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "print", "range",
        "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "isinstance", "type", "Exception", "ValueError", "KeyError",
        "TypeError", "IndexError", "StopIteration", "True", "False", "None",
    )
}


@dataclass
class SandboxResult:
    success: bool
    stdout: str = ""
    error: str | None = None
    output_vars: dict[str, Any] = field(default_factory=dict)
    chart_paths: list[str] = field(default_factory=list)


def _set_resource_limits(memory_limit_mb: int) -> None:
    """Best-effort memory cap for the child process (Linux only)."""
    try:
        limit_bytes = memory_limit_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        # Some sandboxed/container environments refuse RLIMIT_AS; execution
        # still runs, just without the hard memory cap.
        pass


def _worker(
    code: str,
    df_bytes: bytes,
    extra_context: dict[str, Any],
    capture_vars: list[str],
    chart_dir: str,
    chart_prefix: str,
    memory_limit_mb: int,
    result_queue: "mp.Queue",
) -> None:
    _set_resource_limits(memory_limit_mb)

    import pickle

    df = pickle.loads(df_bytes)

    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df.copy(deep=True),  # copy-on-inject: generated code can never mutate the caller's df
    }
    namespace.update(extra_context)

    stdout_buf = io.StringIO()
    chart_paths: list[str] = []
    try:
        with redirect_stdout(stdout_buf):
            exec(code, namespace)  # noqa: S102 - this is the sandbox's whole job

        output_vars = {}
        for name in capture_vars:
            if name in namespace:
                try:
                    pickle.dumps(namespace[name])  # only return what's actually picklable
                    output_vars[name] = namespace[name]
                except Exception:
                    output_vars[name] = repr(namespace[name])

        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            path = f"{chart_dir}/{chart_prefix}_{fig_num}.png"
            fig.savefig(path, bbox_inches="tight", dpi=110)
            chart_paths.append(path)
        plt.close("all")

        result_queue.put(
            {
                "success": True,
                "stdout": stdout_buf.getvalue(),
                "error": None,
                "output_vars": output_vars,
                "chart_paths": chart_paths,
            }
        )
    except Exception:
        plt.close("all")
        result_queue.put(
            {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "error": traceback.format_exc(),
                "output_vars": {},
                "chart_paths": [],
            }
        )


def run_sandboxed(
    code: str,
    df: pd.DataFrame,
    extra_context: dict[str, Any] | None = None,
    capture_vars: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
    chart_dir: str = "outputs/charts",
    chart_prefix: str = "chart",
) -> SandboxResult:
    """Execute `code` in an isolated child process against a copy of `df`.

    `extra_context` may carry small, already-aggregated values (e.g. a
    findings list from a prior stage) into the namespace — never raw
    unaggregated data beyond `df` itself.
    """
    import pickle

    # "spawn" (not "fork"): the host process (Streamlit's server, the eval
    # harness) is multi-threaded, and forking a multi-threaded process risks
    # deadlocking the child on a lock held by another thread at fork time.
    # spawn is slower to start but safe. Callers must invoke run_sandboxed()
    # from inside a function, not bare module-level code, so the child's
    # re-import of __main__ doesn't re-trigger the call.
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    df_bytes = pickle.dumps(df)

    proc = ctx.Process(
        target=_worker,
        args=(
            code,
            df_bytes,
            extra_context or {},
            capture_vars or [],
            chart_dir,
            chart_prefix,
            memory_limit_mb,
            result_queue,
        ),
    )
    proc.start()

    # Must read from the queue *before* joining, not after: multiprocessing's
    # Queue writes to its underlying pipe from a background feeder thread in
    # the child, and a child that put()s a large object (e.g. a captured
    # DataFrame) cannot exit until that thread finishes writing. If the
    # pickled payload exceeds the OS pipe buffer (~64KB on Linux) the write
    # blocks until the parent drains it — so join()-before-get() deadlocks
    # the parent waiting for an exit that can't happen until the parent
    # itself reads the queue. (Confirmed live: a `df` capture on a ~6k-row
    # dataframe pickles to ~560KB and reliably hung the old join-first
    # ordering for the full timeout on every attempt, despite the child
    # finishing its actual work in under 30ms — see README "Key Learnings".)
    try:
        raw = result_queue.get(timeout=timeout)
    except queue.Empty:
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return SandboxResult(
                success=False,
                error=f"TimeoutError: execution exceeded {timeout}s and was terminated.",
            )
        # Process exited without ever putting a result (e.g. OOM-killed by
        # the memory limit, or a segfault in a C extension).
        exit_code = proc.exitcode
        return SandboxResult(
            success=False,
            error=(
                f"Sandbox process exited (code {exit_code}) without returning a "
                "result — likely hit the memory limit or crashed."
            ),
        )

    proc.join(5)  # queue is drained, so the child's feeder thread can finish; this should be near-instant
    if proc.is_alive():
        proc.terminate()
        proc.join()

    return SandboxResult(
        success=raw["success"],
        stdout=raw["stdout"],
        error=raw["error"],
        output_vars=raw["output_vars"],
        chart_paths=raw["chart_paths"],
    )
