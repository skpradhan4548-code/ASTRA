"""
FastAPI router for the DocuAgent agentic layer.

Mount in your main app:
    from agent_router import router as agent_router
    app.include_router(agent_router)

Endpoints:
    POST /agent/run       — blocking; returns final answer + full transcript JSON
    GET  /agent/stream    — streaming SSE; yields tool calls live as they happen
    GET  /agent/workspace — lists files currently in the agent_workspace sandbox
"""

import os
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.core import run_agent_claude, run_agent_openai, stream_agent_claude
from agent.tools import WORKSPACE

router = APIRouter(prefix="/agent", tags=["agent"])


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    goal: str
    provider: str = "claude"   # "claude" | "openai"
    model: str | None = None   # override the default model if desired


# ─────────────────────────────────────────────────────────────────────────────
# POST /agent/run  — blocking, full JSON response
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run")
def run_agent(req: AgentRequest):
    kwargs = {}
    if req.model:
        kwargs["model"] = req.model

    if req.provider == "openai":
        return run_agent_openai(req.goal, **kwargs)

    return run_agent_claude(req.goal, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# GET /agent/stream  — server-sent events, live tool-call feed
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stream")
def stream_agent(
    goal: str = Query(..., description="The goal for the agent to accomplish"),
    provider: str = Query("claude", description="Model provider: claude or openai"),
):
    """
    Stream agent execution as server-sent events.
    The client receives one SSE per tool_start / tool_result / answer / error.

    Example (JavaScript):
        const src = new EventSource(`/agent/stream?goal=your+goal+here`);
        src.addEventListener("tool_start", e => console.log(JSON.parse(e.data)));
        src.addEventListener("answer",     e => console.log(JSON.parse(e.data)));
    """
    def event_generator():
        yield from stream_agent_claude(goal)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering if behind a proxy
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /agent/workspace  — inspect sandbox contents
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/workspace")
def list_workspace():
    """List all files in the agent_workspace sandbox directory."""
    files = [
        {
            "path": str(p.relative_to(WORKSPACE)),
            "size_bytes": p.stat().st_size,
        }
        for p in WORKSPACE.rglob("*")
        if p.is_file()
    ]
    return {"workspace": str(WORKSPACE), "files": files}
