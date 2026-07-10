"""In-process job event bus for streaming agent progress to the browser.

A "job" is one background ingestion (plus its auto-compile). Agent code is
untouched: a logging.Handler captures every log record emitted by pkms.*
loggers while the job's thread is bound via a ContextVar, and pushes the
formatted lines onto the job's queue. An SSE endpoint drains the queue.
"""

import logging
import queue
import threading
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_current_job: ContextVar[str | None] = ContextVar("pkms_current_job", default=None)

JOB_TTL_SECONDS = 900          # jobs with no activity (log lines) for this long are
                               # considered abandoned and evicted — a slow but alive
                               # job keeps itself resident by logging progress
DONE_GRACE_SECONDS = 120       # finished jobs linger this long (re-attach window)
STREAM_MAX_SECONDS = 900       # an SSE connection never outlives this
KEEPALIVE_SECONDS = 15.0       # comment frame cadence while the queue is idle


class _JobState:
    def __init__(self, target: str = "") -> None:
        self.queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.created = time.monotonic()
        self.last_activity = self.created
        self.target = target
        self.status = "queued"         # "queued" | "running" | "done"
        self.result_html = ""
        self.done_at: float | None = None


_jobs: dict[str, _JobState] = {}
_jobs_lock = threading.Lock()


def _evict_stale() -> None:
    """Drop finished jobs past their grace window and abandoned live ones.

    A running job is evicted on *inactivity*, not age: rate-limited compiles
    legitimately exceed any fixed wall-clock TTL, and evicting them mid-flight
    turns publish/finish into silent no-ops (the browser gets an empty "done"
    while the work continues). Must be called with _jobs_lock held.
    """
    now = time.monotonic()
    for job_id, s in list(_jobs.items()):
        if s.status == "done":
            if s.done_at is not None and now - s.done_at > DONE_GRACE_SECONDS:
                _jobs.pop(job_id, None)
        elif now - s.last_activity > JOB_TTL_SECONDS:
            _jobs.pop(job_id, None)


def create_job(target: str = "") -> str:
    """Register a new job and return its id. target is a display label."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _evict_stale()
        _jobs[job_id] = _JobState(target)
    return job_id


def mark_running(job_id: str) -> None:
    """Flip a queued job to running (called when a worker picks it up)."""
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is not None and state.status == "queued":
            state.status = "running"
            state.last_activity = time.monotonic()


def list_active() -> list[dict[str, str]]:
    """Return queued + running jobs (for the Ingest page to re-attach to)."""
    with _jobs_lock:
        _evict_stale()
        return [
            {"job_id": jid, "target": s.target, "status": s.status}
            for jid, s in _jobs.items()
            if s.status in ("queued", "running")
        ]


def bind_job(job_id: str) -> Token:
    """Bind the calling thread to a job; log records now stream to it."""
    return _current_job.set(job_id)


def unbind_job(token: Token) -> None:
    _current_job.reset(token)


def publish(job_id: str, line: str) -> None:
    """Push a log line onto a job's queue (no-op if the job is gone)."""
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is not None:
            state.last_activity = time.monotonic()
            state.queue.put(("log", line))


def finish(job_id: str, result_html: str = "") -> None:
    """Mark a job complete; result_html is delivered as the final event.

    The job is kept in the registry (status="done") for a grace window so a
    browser that navigated away and came back can still re-attach and see the
    result; eviction is handled by _evict_stale.
    """
    with _jobs_lock:
        state = _jobs.get(job_id)
        if state is not None:
            state.status = "done"
            state.result_html = result_html
            state.done_at = time.monotonic()
            state.queue.put(("done", result_html))


def stream(job_id: str) -> Iterator[tuple[str, str]]:
    """Yield ("log", line) events until ("done", result_html).

    Yields ("ping", "") while idle so the SSE connection stays alive. Unknown job
    ids complete immediately (browser reconnects after a server restart). The job
    is NOT evicted when the connection drops, so navigating away and back can
    re-attach; eviction is left to _evict_stale (TTL / done-grace).
    """
    with _jobs_lock:
        state = _jobs.get(job_id)
    if state is None:
        yield ("done", "")
        return

    # Re-attach to an already-finished job: flush any buffered lines, then deliver
    # the stored result without blocking on the (idle) queue.
    if state.status == "done":
        while True:
            try:
                kind, payload = state.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                yield ("done", payload)
                return
            yield (kind, payload)
        yield ("done", state.result_html)
        return

    deadline = time.monotonic() + STREAM_MAX_SECONDS
    while time.monotonic() < deadline:
        try:
            kind, payload = state.queue.get(timeout=KEEPALIVE_SECONDS)
        except queue.Empty:
            yield ("ping", "")
            continue
        yield (kind, payload)
        if kind == "done":
            return
    yield ("done", "")


# ── logging bridge ────────────────────────────────────────────────────────────

class JobLogHandler(logging.Handler):
    """Routes log records from a job-bound thread onto that job's queue."""

    def emit(self, record: logging.LogRecord) -> None:
        job_id = _current_job.get()
        if job_id is None:
            return
        try:
            publish(job_id, self.format(record))
        except Exception:
            self.handleError(record)


def install_handler() -> None:
    """Attach the job handler to the root "pkms" logger (idempotent)."""
    pkms_logger = logging.getLogger("pkms")
    if any(isinstance(h, JobLogHandler) for h in pkms_logger.handlers):
        return
    handler = JobLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    pkms_logger.addHandler(handler)
    if pkms_logger.level in (logging.NOTSET, 0) or pkms_logger.level > logging.INFO:
        pkms_logger.setLevel(logging.INFO)


def run_job(job_id: str, work: Any, render_result: Any) -> None:
    """Execute work() bound to job_id; deliver render_result(result|exc) as done.

    work: () -> result dict. render_result: (result, error) -> html str.
    Intended as a threading.Thread target.
    """
    mark_running(job_id)  # was "queued" while waiting for a worker slot
    token = bind_job(job_id)
    result, error = None, None
    try:
        result = work()
    except Exception as exc:  # delivered to the browser, never lost
        logger.error("Job failed: %s", exc)
        error = exc
    finally:
        unbind_job(token)
    try:
        html = render_result(result, error)
    except Exception as exc:
        html = f"<div class=\"alert alert-error\"><strong>Error:</strong> render failed: {exc}</div>"
    finish(job_id, html)
