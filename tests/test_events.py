import logging
import threading
import time

import pytest

import pkms.events as events
from pkms.events import (
    JobLogHandler,
    bind_job,
    create_job,
    finish,
    install_handler,
    publish,
    run_job,
    stream,
    unbind_job,
)


def _drain(job_id):
    return list(stream(job_id))


# ── queue basics ──────────────────────────────────────────────────────────────

def test_publish_then_finish_streams_in_order():
    job = create_job()
    publish(job, "step 1")
    publish(job, "step 2")
    finish(job, "<div>done</div>")
    got = _drain(job)
    assert got == [("log", "step 1"), ("log", "step 2"), ("done", "<div>done</div>")]


def test_stream_unknown_job_completes_immediately():
    assert _drain("nonexistent") == [("done", "")]


def test_done_job_lingers_for_reattach():
    # A finished job is kept (status "done") so a returning browser can re-attach.
    job = create_job()
    finish(job, "<div>result</div>")
    _drain(job)
    assert job in events._jobs
    assert events._jobs[job].status == "done"


def test_reattach_to_done_job_redelivers_result():
    # Second stream() on an already-drained, finished job still yields the result.
    job = create_job()
    finish(job, "<div>final</div>")
    _drain(job)                      # first consumer drains the queue
    again = _drain(job)              # re-attach after "navigating away and back"
    assert again == [("done", "<div>final</div>")]


def test_done_job_evicted_after_grace():
    job = create_job()
    finish(job, "")
    events._jobs[job].done_at -= events.DONE_GRACE_SECONDS + 1
    create_job()                     # triggers _evict_stale
    assert job not in events._jobs


def test_list_active_lists_active_excludes_done():
    active_job = create_job(target="paper.pdf → demo")
    done = create_job(target="other.pdf → demo")
    finish(done, "")
    active = events.list_active()
    ids = {j["job_id"] for j in active}
    assert active_job in ids
    assert done not in ids
    entry = next(j for j in active if j["job_id"] == active_job)
    assert entry["target"] == "paper.pdf → demo"
    assert entry["status"] == "queued"   # not yet picked up by a worker


def test_create_job_stores_target():
    job = create_job(target="x.pdf → proj")
    assert events._jobs[job].target == "x.pdf → proj"


def test_create_job_starts_queued_and_mark_running():
    job = create_job()
    assert events._jobs[job].status == "queued"
    assert job in {j["job_id"] for j in events.list_active()}  # queued shows as active
    events.mark_running(job)
    assert events._jobs[job].status == "running"


def test_executor_serializes_jobs_and_status_reflects_queue():
    # With a single worker, a second submitted job stays "queued" until the
    # first finishes — proving ingests are serialized, not run concurrently.
    from concurrent.futures import ThreadPoolExecutor
    gate = threading.Event()
    started = threading.Event()
    j1 = create_job("a")
    j2 = create_job("b")

    def work1():
        started.set()
        gate.wait(2)
        return {}

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        ex.submit(run_job, j1, work1, lambda r, e: "")
        ex.submit(run_job, j2, lambda: {}, lambda r, e: "")
        assert started.wait(2)
        assert events._jobs[j1].status == "running"
        assert events._jobs[j2].status == "queued"   # waiting for the one worker
        gate.set()
    finally:
        ex.shutdown(wait=True)
    assert events._jobs[j1].status == "done"
    assert events._jobs[j2].status == "done"


def test_publish_to_finished_job_is_noop():
    job = create_job()
    finish(job, "")
    _drain(job)
    publish(job, "late line")  # must not raise


def test_stale_jobs_evicted_on_create(monkeypatch):
    # Abandoned = no activity for the TTL, whatever the status.
    job = create_job()
    events._jobs[job].last_activity -= events.JOB_TTL_SECONDS + 1
    create_job()
    assert job not in events._jobs


def test_running_job_with_recent_activity_survives_ttl():
    # A slow-but-alive job (e.g. a rate-limited compile) must NOT be evicted:
    # it keeps itself resident by logging progress via publish().
    job = create_job()
    events.mark_running(job)
    events._jobs[job].created -= events.JOB_TTL_SECONDS + 1       # older than TTL...
    events._jobs[job].last_activity -= events.JOB_TTL_SECONDS + 1
    publish(job, "still compiling")                               # ...but active now
    create_job()                     # triggers _evict_stale
    assert job in events._jobs
    assert events._jobs[job].status == "running"


# ── logging bridge ────────────────────────────────────────────────────────────

def test_handler_routes_bound_thread_logs_only():
    install_handler()
    job = create_job()
    test_logger = logging.getLogger("pkms.test_events")

    def bound_work():
        token = bind_job(job)
        test_logger.info("from bound thread")
        unbind_job(token)
        finish(job, "")

    test_logger.info("unbound line — must not stream")
    t = threading.Thread(target=bound_work)
    t.start()
    t.join()

    got = _drain(job)
    assert ("log", "from bound thread") in got
    assert all("unbound" not in payload for _, payload in got)


def test_install_handler_idempotent():
    install_handler()
    install_handler()
    handlers = [h for h in logging.getLogger("pkms").handlers if isinstance(h, JobLogHandler)]
    assert len(handlers) == 1


# ── run_job ───────────────────────────────────────────────────────────────────

def test_run_job_delivers_rendered_result():
    job = create_job()
    run_job(job, lambda: {"status": "DONE"},
            lambda result, error: f"<div>{result['status']}</div>")
    got = _drain(job)
    assert got[-1] == ("done", "<div>DONE</div>")


def test_run_job_delivers_error():
    job = create_job()
    run_job(job, lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            lambda result, error: f"<div>err: {error}</div>")
    got = _drain(job)
    assert got[-1] == ("done", "<div>err: boom</div>")
    # the failure was also logged into the stream
    assert any("boom" in payload for kind, payload in got if kind == "log")
