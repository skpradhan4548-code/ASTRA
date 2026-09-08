"""
Provider-agnostic agentic tool-use loop.

Two transports are provided:
  run_agent_claude()  — Anthropic Messages API (default; Claude Sonnet)
  run_agent_openai()  — OpenAI Chat Completions API (swap in gpt-4o or Astra)

Both share the same tool registry (agent.tools.TOOLS) and the same
architectural pattern:
  1. Send user goal + available tools to the model
  2. Model replies with a final answer OR one or more tool calls
  3. If tool calls: execute them locally, send results back
  4. Repeat until the model returns a final answer or MAX_TURNS is hit

Streaming variant:
  stream_agent_claude() — generator that yields SSE-formatted JSON events,
  one per tool call or final answer. Mount behind StreamingResponse in
  agent_router.py for a live transcript in the UI.
"""

import json
import os
from typing import Generator

from agent.tools import TOOLS

MAX_TURNS = 10

SYSTEM_PROMPT = """\
You are DocuAgent, an intelligent research and automation assistant.
You have access to a set of tools and should use them proactively to \
complete the user's goal step by step.

Tool-use strategy:
1. If the goal involves information that might already be in the local \
document store, call search_documents FIRST.
2. If that returns nothing useful, fall back to web_search.
3. If the goal involves computation, use run_python.
4. If the goal involves reading or writing files, use read_file / write_file.
5. If the goal involves analysing tabular data, use analyze_dataframe.
6. When you have enough information, give a clear, concise final answer.

Always be transparent about which tool you used and why.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [truncated, {len(text)} chars total]"


def _sse(event: str, data: dict) -> str:
    """Format a server-sent event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic (Claude) — blocking
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_claude(user_goal: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool_specs = [t["spec"] for t in TOOLS.values()]

    messages = [{"role": "user", "content": user_goal}]
    transcript = []

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tool_specs,
            messages=messages,
        )
        transcript.append({"turn": turn, "stop_reason": response.stop_reason})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {
                "answer": final_text,
                "turns": turn + 1,
                "transcript": transcript,
            }

        # Model wants to call tools — execute each, collect results, loop
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOLS[block.name]["fn"]
            try:
                result = tool_fn(**block.input)
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            transcript.append({
                "type": "tool_call",
                "tool": block.name,
                "input": block.input,
                "result_preview": _truncate(result),
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": None,
        "error": "max_turns reached without a final answer",
        "transcript": transcript,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic (Claude) — streaming (yields SSE strings)
# ─────────────────────────────────────────────────────────────────────────────

def stream_agent_claude(
    user_goal: str, model: str = "claude-sonnet-4-6"
) -> Generator[str, None, None]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool_specs = [t["spec"] for t in TOOLS.values()]

    messages = [{"role": "user", "content": user_goal}]

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tool_specs,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            yield _sse("answer", {"text": final_text, "turns": turn + 1})
            yield _sse("done", {})
            return

        # Execute tool calls and stream each one as an event
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            yield _sse("tool_start", {"tool": block.name, "input": block.input})
            tool_fn = TOOLS[block.name]["fn"]
            try:
                result = tool_fn(**block.input)
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            yield _sse("tool_result", {
                "tool": block.name,
                "result_preview": _truncate(result),
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    yield _sse("error", {"message": "max_turns reached without a final answer"})
    yield _sse("done", {})


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI transport — same tool registry, different wire format
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_openai(user_goal: str, model: str = "gpt-4o") -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": t["spec"]["name"],
                "description": t["spec"]["description"],
                "parameters": t["spec"]["input_schema"],
            },
        }
        for t in TOOLS.values()
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_goal},
    ]
    transcript = []

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=tool_specs,
            messages=messages,
        )
        msg = response.choices[0].message
        transcript.append({"turn": turn, "finish_reason": response.choices[0].finish_reason})

        if not msg.tool_calls:
            return {"answer": msg.content, "turns": turn + 1, "transcript": transcript}

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": msg.tool_calls,
        })
        for call in msg.tool_calls:
            tool_fn = TOOLS[call.function.name]["fn"]
            args = json.loads(call.function.arguments)
            try:
                result = tool_fn(**args)
            except Exception as exc:
                result = json.dumps({"error": str(exc)})
            transcript.append({
                "type": "tool_call",
                "tool": call.function.name,
                "input": args,
                "result_preview": _truncate(result),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return {
        "answer": None,
        "error": "max_turns reached without a final answer",
        "transcript": transcript,
    }
