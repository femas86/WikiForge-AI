"""Background-job / SSE routes (roadmap 1.12 split). Live-ingest stream + the
re-attach panel for in-progress jobs. Imported last by pkms.web so the shared
infra (templates) is in place."""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from markupsafe import escape

from pkms.events import astream, list_active
from pkms.web import templates

router = APIRouter()


@router.get("/ui/ingest/active", response_class=HTMLResponse)
def ui_ingest_active(request: Request):
    """Fragment listing in-progress ingest jobs so the Ingest page can re-attach
    to their live streams after the user navigated away and came back."""
    return templates.TemplateResponse(request, "ingest_active.html", {
        "jobs": list_active(),
    })


@router.get("/ui/ingest/stream/{job_id}")
async def ui_ingest_stream(job_id: str):
    """SSE stream of one ingest job's log lines; closes with the result card.

    async so an open stream costs an event-loop task, not a pinned threadpool
    worker (see events.astream)."""
    async def event_source():
        async for kind, payload in astream(job_id):
            if kind == "ping":
                yield ": keep-alive\n\n"
            elif kind == "log":
                yield f"event: log\ndata: <div>{escape(payload)}</div>\n\n"
            else:  # done — multi-line HTML payload
                data = "".join(f"data: {line}\n" for line in payload.splitlines()) or "data: \n"
                yield f"event: done\n{data}\n"
                yield "\n"
    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
