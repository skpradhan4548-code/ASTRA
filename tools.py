"""
Tool implementations for the DocuAgent agentic layer.

Each tool is a plain Python function plus a JSON-schema "spec" dict that
gets handed to the LLM (Claude or Astra) so it knows the tool exists and
how to call it. Add new tools by writing a function + spec, then
registering both in TOOLS at the bottom of this file.
"""

import io
import json
import subprocess
import contextlib
from pathlib import Path

import pandas as pd
import requests

# Restrict file tools to a sandbox directory so the agent can't touch
# anything outside your project's working folder.
WORKSPACE = Path("./agent_workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)


def _safe_path(relative_path: str) -> Path:
    """Resolve a path inside WORKSPACE only — blocks path traversal."""
    p = (WORKSPACE / relative_path).resolve()
    if WORKSPACE not in p.parents and p != WORKSPACE:
        raise ValueError(f"Path {relative_path!r} escapes the workspace sandbox")
    return p


# ---------------------------------------------------------------------
# Tool: web_search  (stand-in for "browsing" — swap in a real search API)
# ---------------------------------------------------------------------
def web_search(query: str) -> str:
    """
    Minimal web search using DuckDuckGo's HTML endpoint (no API key needed
    for a demo). For production, swap in Tavily, Serper, or Bing Search API.
    """
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        # Extremely light scrape — good enough for a demo, not production.
        import re
        titles = re.findall(r'class="result__a"[^>]*>(.*?)</a>', resp.text)
        clean = [re.sub("<.*?>", "", t) for t in titles[:5]]
        return json.dumps({"query": query, "top_results": clean}) if clean else \
            json.dumps({"query": query, "top_results": [], "note": "no results parsed"})
    except Exception as e:
        return json.dumps({"error": str(e)})


WEB_SEARCH_SPEC = {
    "name": "web_search",
    "description": "Search the web and return top result titles for a query.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------
# Tool: run_python  (stand-in for "software engineering" / code execution)
# ---------------------------------------------------------------------
def run_python(code: str, timeout_seconds: int = 10) -> str:
    """
    Executes short Python snippets in a subprocess with a timeout.
    NOTE: This is a demo sandbox, not a security boundary. For anything
    user-facing or multi-tenant, run this in a locked-down container
    (gVisor, firecracker-vm, or similar) — never bare subprocess in prod.
    """
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return json.dumps({
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
            "exit_code": result.returncode,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"execution exceeded {timeout_seconds}s"})


RUN_PYTHON_SPEC = {
    "name": "run_python",
    "description": "Execute a short Python snippet and return stdout/stderr. Use for calculations, data transforms, or quick checks.",
    "input_schema": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}


# ---------------------------------------------------------------------
# Tool: read_file / write_file  (stand-in for "office/document automation")
# ---------------------------------------------------------------------
def read_file(relative_path: str) -> str:
    p = _safe_path(relative_path)
    if not p.exists():
        return json.dumps({"error": f"{relative_path} not found in workspace"})
    if p.suffix in (".csv",):
        df = pd.read_csv(p)
        return df.head(50).to_json(orient="records")
    if p.suffix in (".xlsx",):
        df = pd.read_excel(p)
        return df.head(50).to_json(orient="records")
    return p.read_text(errors="replace")[:8000]


READ_FILE_SPEC = {
    "name": "read_file",
    "description": "Read a text, CSV, or XLSX file from the sandboxed workspace. Returns text or the first 50 rows as JSON for tabular files.",
    "input_schema": {
        "type": "object",
        "properties": {"relative_path": {"type": "string"}},
        "required": ["relative_path"],
    },
}


def write_file(relative_path: str, content: str) -> str:
    p = _safe_path(relative_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return json.dumps({"written": str(p.relative_to(WORKSPACE)), "bytes": len(content)})


WRITE_FILE_SPEC = {
    "name": "write_file",
    "description": "Write text content to a file inside the sandboxed workspace.",
    "input_schema": {
        "type": "object",
        "properties": {
            "relative_path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["relative_path", "content"],
    },
}


# ---------------------------------------------------------------------
# Tool: analyze_dataframe  (stand-in for "scientific / data analysis")
# ---------------------------------------------------------------------
def analyze_dataframe(relative_path: str, operation: str) -> str:
    """
    operation: one of 'describe', 'columns', 'nulls', 'correlation'
    Kept intentionally small — this is the seed you grow into a real
    data-analysis tool (add groupby, plotting-to-png, etc. as next steps).
    """
    p = _safe_path(relative_path)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)

    if operation == "describe":
        return df.describe(include="all").to_json()
    if operation == "columns":
        return json.dumps(list(df.columns))
    if operation == "nulls":
        return df.isnull().sum().to_json()
    if operation == "correlation":
        return df.corr(numeric_only=True).to_json()
    return json.dumps({"error": f"unknown operation {operation}"})


ANALYZE_DATAFRAME_SPEC = {
    "name": "analyze_dataframe",
    "description": "Run a quick analysis (describe, columns, nulls, correlation) on a CSV/XLSX file in the workspace.",
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


# ---------------------------------------------------------------------
# Registry — the agent loop imports this single dict
# ---------------------------------------------------------------------
TOOLS = {
    "web_search": {"fn": web_search, "spec": WEB_SEARCH_SPEC},
    "run_python": {"fn": run_python, "spec": RUN_PYTHON_SPEC},
    "read_file": {"fn": read_file, "spec": READ_FILE_SPEC},
    "write_file": {"fn": write_file, "spec": WRITE_FILE_SPEC},
    "analyze_dataframe": {"fn": analyze_dataframe, "spec": ANALYZE_DATAFRAME_SPEC},
}
