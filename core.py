"""
Provider-agnostic agentic tool-use loop.

Works with Anthropic's Messages API (Claude) out of the box. A second
function, run_agent_openai(), does the same thing against OpenAI's
Responses API so you can point it at gpt-6-astra once you have access —
the tool specs and tool functions in tools.py are shared between both,
you only swap the transport layer.

This is the same architectural pattern behind "computer use" agents:
  1. Send the user goal + available tools to the model
  2. Model replies with either a final answer OR a tool call
  3. If a tool call: execute it locally, send the result back
  4. Repeat until the model returns a final answer or you hit max_turns
"""

import os
import json
from agent.tools import TOOLS

MAX_TURNS = 8


# ---------------------------------------------------------------------
# Anthropic (Claude) transport — matches your existing stack default
# ---------------------------------------------------------------------
def run_agent_claude(user_goal: str, model: str = "claude-sonnet-4-6") -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    tool_specs = [t["spec"] for t in TOOLS.values()]

    messages = [{"role": "user", "content": user_goal}]
    transcript = []

    for turn in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            tools=tool_specs,
            messages=messages,
        )
        transcript.append({"turn": turn, "stop_reason": response.stop_reason})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "turns": turn + 1, "transcript": transcript}

        # Model wants to call one or more tools — execute each, then loop
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_fn = TOOLS[block.name]["fn"]
            try:
                result = tool_fn(**block.input)
            except Exception as e:
                result = json.dumps({"error": str(e)})
            transcript.append({"tool": block.name, "input": block.input, "result": result[:500]})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        messages.append({"role": "user", "content": tool_results})

    return {"answer": None, "error": "max_turns reached without a final answer", "transcript": transcript}


# ---------------------------------------------------------------------
# OpenAI (gpt-6-astra) transport — same tool registry, different wire format
# ---------------------------------------------------------------------
def run_agent_openai(user_goal: str, model: str = "gpt-6-astra") -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # OpenAI's function-calling schema wraps the same spec shape slightly differently
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

    messages = [{"role": "user", "content": user_goal}]
    transcript = []

    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=model,
            max_tokens=1500,
            tools=tool_specs,
            messages=messages,
        )
        msg = response.choices[0].message
        transcript.append({"turn": turn, "finish_reason": response.choices[0].finish_reason})

        if not msg.tool_calls:
            return {"answer": msg.content, "turns": turn + 1, "transcript": transcript}

        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
        for call in msg.tool_calls:
            tool_fn = TOOLS[call.function.name]["fn"]
            args = json.loads(call.function.arguments)
            try:
                result = tool_fn(**args)
            except Exception as e:
                result = json.dumps({"error": str(e)})
            transcript.append({"tool": call.function.name, "input": args, "result": result[:500]})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

    return {"answer": None, "error": "max_turns reached without a final answer", "transcript": transcript}
