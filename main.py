"""
DocuAgent Agentic Layer — FastAPI entry point.

Run with:
    uvicorn main:app --reload --port 8000

Then open http://localhost:8000 to see the interactive demo UI.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env before importing anything that reads env vars
load_dotenv()

from agent_router import router as agent_router  # noqa: E402 (after load_dotenv)

app = FastAPI(
    title="DocuAgent — Agentic Tool-Use Layer",
    description=(
        "An agentic loop on top of a RAG pipeline. "
        "The agent can search the web, run code, read/write files, "
        "analyze data, and retrieve from a local document store."
    ),
    version="1.0.0",
)

# Allow cross-origin requests (useful when the frontend is served separately during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the agent API routes at /agent/*
app.include_router(agent_router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "anthropic_key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "chroma_collection": os.environ.get("CHROMA_COLLECTION", "docuagent"),
    }


# Serve the static frontend — must come AFTER all API routes
FRONTEND = Path(__file__).parent / "frontend"
if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

