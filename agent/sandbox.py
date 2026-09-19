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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Overridable via env vars so hosts with unusual constraints can tune
# without a code change. Defaults now assume a container-limited deploy
# (Streamlit Cloud, Cloud Run, etc.): SANDBOX_MEMORY_LIMIT_MB=0 disables
# the RLIMIT_AS-based cap and lets the container's own memory limits do
# their job — see _set_resource_limits() for why RLIMIT_AS was making
# things strictly worse there. On a local dev machine with no container
# cap, set a real number (e.g. 1024) if you want a per-snippet safety
# net; the container backstop isn't there.
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", 15))
DEFAULT_MEMORY_LIMIT_MB = int(os.environ.get("SANDBOX_MEMORY_LIMIT_MB", 0))

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
    chart_paths: list[str] = field(default_factory=list)  # interactive standalone .html files
    # Optional static .png per chart (same order/index as chart_paths), only
    # populated when the `kaleido` package is installed — see _worker() for
    # why that's an optional dependency, not a hard one. "" where unavailable.
    static_chart_paths: list[str] = field(default_factory=list)


def _set_resource_limits(memory_limit_mb: int) -> None:
    """Best-effort memory cap for the child process (Linux only).

    Pass 0 to skip the cap entirely — the right choice on container-limited
    hosts (Streamlit Cloud, Cloud Run, most PaaS) where the container
    already enforces total memory. RLIMIT_AS is a poor proxy for actual
    memory usage: it counts memory-mapped shared libraries (pandas alone
    mmaps hundreds of MB of native code), thread stacks (~8MB each in
    virtual address space), and other things the process isn't really
    "using". A limit that looks generous on a dev machine (1GB) can starve
    the pandas/numpy/matplotlib import in the child before any user code
    runs — and worse, once address space is that tight, `Queue.put()`
    itself fails (it needs to spawn a background feeder thread whose stack
    doesn't fit), so the error handler that would report the MemoryError
    can't even run. That leaves the parent seeing an opaque "process
    exited" with zero diagnostic content — the exact failure that kept
    recurring on the deployed free-tier app.
    """
    if memory_limit_mb <= 0:
        return
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
    # Everything below — including setting the resource limit, unpickling
    # the input df, and building the namespace — must be inside this
    # try/except, not just the exec() call. An uncaught exception anywhere
    # in a multiprocessing.Process target (e.g. a MemoryError from
    # pickle.loads() under a tight RLIMIT_AS on a memory-constrained host)
    # crashes the child silently: Python prints a traceback to the child's
    # own stderr and exits with code 1, but nothing is ever put on
    # result_queue, so the parent just sees "process exited without a
    # result" with no diagnostic content. That's the failure mode this
    # guards against — observed live on a free-tier deployment where the
    # host's actual available memory was tighter than local testing showed
    # (see README "Key Learnings").
    stdout_buf = io.StringIO()
    chart_paths: list[str] = []
    try:
        _set_resource_limits(memory_limit_mb)

        import pickle

        df = pickle.loads(df_bytes)

        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd,
            "np": np,
            "px": px,
            "go": go,
            "df": df.copy(deep=True),  # copy-on-inject: generated code can never mutate the caller's df
        }
        namespace.update(extra_context)

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

        # Plotly has no global figure registry the way matplotlib's pyplot
        # does (there was no plt.get_fignums() equivalent to fall back on),
        # so the chart-building prompt (agent/stages/chart.py) asks the
        # model to collect its own figures into a plain list named `charts`
        # — the worker just reads that convention back out of the
        # namespace, same shape as any other captured variable.
        static_chart_paths: list[str] = []
        charts_obj = namespace.get("charts")
        if isinstance(charts_obj, list):
            for i, fig in enumerate(charts_obj):
                if not hasattr(fig, "write_html"):
                    continue
                html_path = f"{chart_dir}/{chart_prefix}_{i}.html"
                try:
                    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
                    chart_paths.append(html_path)
                except Exception:
                    continue
                # Static PNG export needs the optional `kaleido` package,
                # which bundles its own renderer — deliberately NOT a hard
                # dependency (see requirements.txt) since this project
                # deploys to a ~1GB container where an extra native
                # dependency is a real memory-budget decision, not a free
                # one. Runs inside this already-isolated, already
                # resource-limited child process either way, so a failure
                # or absence here can never affect the interactive chart
                # the app actually renders — it only means the MCP server
                # falls back to a text note instead of a static image.
                try:
                    png_path = f"{chart_dir}/{chart_prefix}_{i}.png"
                    fig.write_image(png_path)
                    static_chart_paths.append(png_path)
                except Exception:
                    static_chart_paths.append("")

        result_queue.put(
            {
                "success": True,
                "stdout": stdout_buf.getvalue(),
                "error": None,
                "output_vars": output_vars,
                "chart_paths": chart_paths,
                "static_chart_paths": static_chart_paths,
            }
        )
    except Exception:
        result_queue.put(
            {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "error": traceback.format_exc(),
                "output_vars": {},
                "chart_paths": [],
                "static_chart_paths": [],
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
    #
    # Poll rather than a straight blocking get: on a silent crash (OOM
    # kill, or a MemoryError so severe the child's own exception handler
    # can't run), a bare `queue.get(timeout=15)` waits the FULL 15s before
    # noticing the child died in 100ms, wasting the parent's time and
    # producing pipelines that take ~45s to fail three stages. Poll every
    # 200ms so we notice a dead child roughly one poll interval after it
    # dies, whether it produced output or not.
    import time as _time
    poll_interval = 0.2
    deadline = _time.monotonic() + timeout
    raw = None
    while True:
        try:
            raw = result_queue.get(timeout=poll_interval)
            break
        except queue.Empty:
            if not proc.is_alive():
                # Give the queue one last poll: the child may have put a
                # result and exited in the same interval.
                try:
                    raw = result_queue.get_nowait()
                    break
                except queue.Empty:
                    exit_code = proc.exitcode
                    return SandboxResult(
                        success=False,
                        error=(
                            f"Sandbox process exited (code {exit_code}) without returning a "
                            "result — likely OOM-killed by the host container or crashed "
                            "in a C extension. If this is a deployed instance on a "
                            "memory-constrained host, unset SANDBOX_MEMORY_LIMIT_MB (or set "
                            "it to 0) so RLIMIT_AS isn't fighting the container's own cap."
                        ),
                    )
            if _time.monotonic() >= deadline:
                proc.terminate()
                proc.join(2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                return SandboxResult(
                    success=False,
                    error=f"TimeoutError: execution exceeded {timeout}s and was terminated.",
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
        static_chart_paths=raw.get("static_chart_paths", []),
    )
