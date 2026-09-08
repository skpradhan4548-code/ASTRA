"""
Tool implementations for the DocuAgent agentic layer.

Each tool is a plain Python function plus a JSON-schema "spec" dict that
gets handed to the LLM (Claude or OpenAI) so it knows the tool exists and
how to call it.

Tools registered here:
  web_search          — DuckDuckGo scrape; swap in Tavily/Serper for prod
  run_python          — subprocess code execution (demo sandbox only)
  read_file           — read text/CSV/XLSX from sandboxed workspace
  write_file          — write text to sandboxed workspace
  analyze_dataframe   — describe/columns/nulls/correlation on CSV or XLSX
  list_workspace_files — list all files currently in the sandbox
  search_documents    — ChromaDB RAG retrieval (graceful no-op if not configured)

To add a new tool:
  1. Write a Python function.
  2. Write a spec dict (JSON-schema "input_schema", name, description).
  3. Register both in the TOOLS dict at the bottom of this file.
"""

import io
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests

from agent.sandbox import NetworkPolicy, default_sandbox

# ---------------------------------------------------------------------------
# Sandbox workspace — agent file tools are restricted to this directory
# ---------------------------------------------------------------------------
WORKSPACE = Path("./agent_workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def _safe_path(relative_path: str, session_id: str | None = None) -> Path:
    """
    Resolve a path inside WORKSPACE only — strictly blocks path traversal,
    null bytes, and symlink escapes.
    Supports session-level subdirectories: agent_workspace/sessions/{session_id}/
    """
    if not relative_path or "\x00" in relative_path:
        raise ValueError("Invalid file path provided.")

    base = WORKSPACE
    if session_id:
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if safe_session:
            base = WORKSPACE / "sessions" / safe_session
            base.mkdir(parents=True, exist_ok=True)

    base = base.resolve()
    candidate = (base / relative_path).resolve()

    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError(f"Path {relative_path!r} escapes the workspace sandbox")

    return candidate


# ---------------------------------------------------------------------------
# Tool: web_search
# ---------------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    Minimal web search using DuckDuckGo's HTML endpoint (no API key needed
    for a demo). For production swap in Tavily, Serper, or Bing Search API.
    """
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        import re
        # Pull titles + snippets for richer results
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</span>', resp.text)
        clean_titles = [re.sub("<.*?>", "", t) for t in titles[:5]]
        clean_snippets = [re.sub("<.*?>", "", s) for s in snippets[:5]]
        results = [
            {"title": t, "snippet": s}
            for t, s in zip(clean_titles, clean_snippets)
        ]
        if results:
            return json.dumps({"query": query, "results": results}, indent=2)
        return json.dumps({"query": query, "results": [], "note": "no results parsed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": (
        "Search the web and return the top result titles and snippets for a query. "
        "Use this when you need current information not in the document store."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."}
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Tool: run_python
# ---------------------------------------------------------------------------
def run_python(
    code: str,
    timeout_seconds: int = 15,
    network: str = "isolated",
    session_id: str | None = None,
) -> str:
    """
    Executes Python snippets inside a hardened sandbox (Docker if operational,
    with automatic fallback to ProcessSandbox) with memory limits, timeouts,
    and network isolation policies.
    """
    net_policy = NetworkPolicy.ISOLATED if network == "isolated" else NetworkPolicy.OPEN
    target_workspace = WORKSPACE
    if session_id:
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if safe_session:
            target_workspace = WORKSPACE / "sessions" / safe_session
            target_workspace.mkdir(parents=True, exist_ok=True)

    result = default_sandbox.execute(
        code=code,
        workspace_dir=target_workspace,
        timeout_seconds=timeout_seconds,
        network_policy=net_policy,
    )
    return result.to_json()


RUN_PYTHON_SPEC = {
    "name": "run_python",
    "description": (
        "Execute a Python snippet in an isolated sandbox with resource constraints. "
        "Returns stdout, stderr, exit code, and execution runtime metadata. "
        "Network access can be set to 'isolated' (no network, default) or 'open'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Valid Python source code to execute."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds to wait before termination (default 15).",
                "default": 15,
            },
            "network": {
                "type": "string",
                "enum": ["isolated", "open"],
                "description": "Network policy: 'isolated' (no internet, default) or 'open'.",
                "default": "isolated",
            },
        },
        "required": ["code"],
    },
}



# ---------------------------------------------------------------------------
# Tool: read_file
# ---------------------------------------------------------------------------
def read_file(relative_path: str, session_id: str | None = None) -> str:
    """Read a text, CSV, or XLSX file from the sandboxed workspace."""
    p = _safe_path(relative_path, session_id=session_id)
    if not p.exists():
        return json.dumps({"error": f"{relative_path!r} not found in workspace"})
    if p.suffix in (".csv",):
        df = pd.read_csv(p)
        return df.head(50).to_json(orient="records", indent=2)
    if p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
        return df.head(50).to_json(orient="records", indent=2)
    return p.read_text(errors="replace")[:8000]


READ_FILE_SPEC = {
    "name": "read_file",
    "description": (
        "Read a text, CSV, or XLSX file from the sandboxed agent workspace. "
        "Returns raw text for plain files, or the first 50 rows as JSON for tabular files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relative_path": {
                "type": "string",
                "description": "Path relative to the agent_workspace directory.",
            }
        },
        "required": ["relative_path"],
    },
}


# ---------------------------------------------------------------------------
# Tool: write_file
# ---------------------------------------------------------------------------
def write_file(relative_path: str, content: str, session_id: str | None = None) -> str:
    """Write text content to a file inside the sandboxed workspace."""
    p = _safe_path(relative_path, session_id=session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return json.dumps(
        {"written": str(p.relative_to(WORKSPACE)), "bytes": len(content)}, indent=2
    )


WRITE_FILE_SPEC = {
    "name": "write_file",
    "description": "Write text content to a named file inside the sandboxed workspace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string", "description": "File path relative to workspace."},
            "content": {"type": "string", "description": "Text content to write."},
        },
        "required": ["relative_path", "content"],
    },
}


# ---------------------------------------------------------------------------
# Tool: analyze_dataframe
# ---------------------------------------------------------------------------
def analyze_dataframe(relative_path: str, operation: str, session_id: str | None = None) -> str:
    """
    Quick statistical analysis on a CSV or XLSX file.
    Operations: describe, columns, nulls, correlation
    """
    p = _safe_path(relative_path, session_id=session_id)
    if not p.exists():
        return json.dumps({"error": f"{relative_path!r} not found in workspace"})
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
    if operation == "describe":
        return df.describe(include="all").to_json(indent=2)
    if operation == "columns":
        return json.dumps(list(df.columns), indent=2)
    if operation == "nulls":
        return df.isnull().sum().to_json(indent=2)
    if operation == "correlation":
        return df.corr(numeric_only=True).to_json(indent=2)
    return json.dumps({"error": f"unknown operation {operation!r}"})



ANALYZE_DATAFRAME_SPEC = {
    "name": "analyze_dataframe",
    "description": (
        "Run a quick statistical analysis on a CSV or XLSX file in the workspace. "
        "Choose operation: 'describe' (summary stats), 'columns' (column names), "
        "'nulls' (missing-value counts), 'correlation' (numeric correlation matrix)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string"},
            "operation": {
                "type": "string",
                "enum": ["describe", "columns", "nulls", "correlation"],
            },
        },
        "required": ["relative_path", "operation"],
    },
}


# ---------------------------------------------------------------------------
# Tool: list_workspace_files
# ---------------------------------------------------------------------------
def list_workspace_files(session_id: str | None = None) -> str:
    """List all files currently in the agent workspace sandbox or active session."""
    base = WORKSPACE
    if session_id:
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if safe_session:
            base = WORKSPACE / "sessions" / safe_session
            base.mkdir(parents=True, exist_ok=True)
    files = [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]
    return json.dumps({"workspace": str(base), "files": files}, indent=2)



LIST_WORKSPACE_FILES_SPEC = {
    "name": "list_workspace_files",
    "description": (
        "List all files currently in the agent_workspace sandbox directory. "
        "Call this before read_file or analyze_dataframe to see what files are available."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool: search_documents  (ChromaDB RAG retrieval)
# ---------------------------------------------------------------------------
def search_documents(query: str, top_k: int = 4) -> str:
    """
    Retrieve the most relevant chunks from the local ChromaDB document store.
    Requires CHROMA_COLLECTION env var and an existing ChromaDB collection.
    Gracefully no-ops with a helpful message if ChromaDB is not configured.
    """
    collection_name = os.environ.get("CHROMA_COLLECTION", "docuagent")
    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(collection_name)
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        hits = [
            {
                "rank": i + 1,
                "score": round(1 - d, 4),  # cosine similarity approximation
                "source": m.get("source", "unknown"),
                "text": doc[:1000],
            }
            for i, (doc, m, d) in enumerate(zip(docs, metas, distances))
        ]
        return json.dumps({"query": query, "hits": hits}, indent=2)

    except ImportError:
        return json.dumps({
            "note": "chromadb not installed — run `pip install chromadb` to enable document search.",
            "query": query,
            "hits": [],
        })
    except Exception as e:
        return json.dumps({
            "note": f"ChromaDB lookup failed: {e}. "
                    "Set CHROMA_COLLECTION and ingest documents first.",
            "query": query,
            "hits": [],
        })


SEARCH_DOCUMENTS_SPEC = {
    "name": "search_documents",
    "description": (
        "Search the local ChromaDB document store and return the most relevant "
        "text chunks for a query. Use this BEFORE web_search when you think the "
        "answer might already be in the uploaded documents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language query to search for."},
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default 4).",
                "default": 4,
            },
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Tool: run_shell
# ---------------------------------------------------------------------------
def run_shell(
    command: str,
    timeout_seconds: int = 20,
    session_id: str | None = None,
) -> str:
    """
    Executes a shell command inside the workspace directory.
    Includes security screening against blocked patterns and process tree termination on timeout.
    """
    target_workspace = WORKSPACE
    if session_id:
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if safe_session:
            target_workspace = WORKSPACE / "sessions" / safe_session
            target_workspace.mkdir(parents=True, exist_ok=True)

    res = default_sandbox.execute_shell(
        command=command,
        workspace_dir=target_workspace,
        timeout_seconds=timeout_seconds,
    )
    return res.to_json()


RUN_SHELL_SPEC = {
    "name": "run_shell",
    "description": (
        "Execute a shell/terminal command inside the sandboxed workspace directory. "
        "Useful for directory inspection, git operations, compiling, or running cli tools. "
        "Dangerous destructive system commands are blocked."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command line to run."},
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds before process tree is terminated (default 20).",
                "default": 20,
            },
        },
        "required": ["command"],
    },
}


# ---------------------------------------------------------------------------
# Tool: install_package
# ---------------------------------------------------------------------------
def install_package(
    package_name: str,
    session_id: str | None = None,
    timeout_seconds: int = 60,
) -> str:
    """
    Installs a Python package into the workspace's dedicated virtual environment.
    Keeps the host environment isolated and clean.
    """
    target_workspace = WORKSPACE
    if session_id:
        safe_session = "".join(c for c in session_id if c.isalnum() or c in ("-", "_")).strip()
        if safe_session:
            target_workspace = WORKSPACE / "sessions" / safe_session
            target_workspace.mkdir(parents=True, exist_ok=True)

    res = default_sandbox.install_package(
        package_name=package_name,
        workspace_dir=target_workspace,
        timeout_seconds=timeout_seconds,
    )
    return res.to_json()


INSTALL_PACKAGE_SPEC = {
    "name": "install_package",
    "description": (
        "Install a Python package (via pip) into the workspace virtual environment. "
        "Use this before running Python code that depends on third-party libraries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "package_name": {
                "type": "string",
                "description": "Package name and optional version specifier (e.g. 'scipy', 'sympy==1.12').",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Max seconds for pip installation (default 60).",
                "default": 60,
            },
        },
        "required": ["package_name"],
    },
}


# ---------------------------------------------------------------------------
# Registry — the agent loop imports this single dict
# ---------------------------------------------------------------------------
TOOLS = {
    "web_search": {"fn": web_search, "spec": WEB_SEARCH_SPEC},
    "run_python": {"fn": run_python, "spec": RUN_PYTHON_SPEC},
    "run_shell": {"fn": run_shell, "spec": RUN_SHELL_SPEC},
    "install_package": {"fn": install_package, "spec": INSTALL_PACKAGE_SPEC},
    "read_file": {"fn": read_file, "spec": READ_FILE_SPEC},
    "write_file": {"fn": write_file, "spec": WRITE_FILE_SPEC},
    "analyze_dataframe": {"fn": analyze_dataframe, "spec": ANALYZE_DATAFRAME_SPEC},
    "list_workspace_files": {"fn": list_workspace_files, "spec": LIST_WORKSPACE_FILES_SPEC},
    "search_documents": {"fn": search_documents, "spec": SEARCH_DOCUMENTS_SPEC},
}

