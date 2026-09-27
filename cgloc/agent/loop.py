from __future__ import annotations

import json
import time
from collections.abc import Callable

from .llm import Reply
from .tools import Tool, call_tool

SYSTEM = """You are a code localization agent. Given a GitHub issue for a Python repository, find the \
functions, methods or classes that must be edited to fix it. You cannot edit code; you only investigate.

Tools:
{tools}
- final(locations): Submit your answer: a list of up to 5 "path::Qualname" strings, most likely first.

Each turn, reply with exactly one JSON object and nothing else:
  {{"thought": "<one short sentence>", "tool": "<tool name>", "args": {{...}}}}
When you are confident, or when told to stop, submit:
  {{"thought": "...", "tool": "final", "args": {{"locations": ["path/to/file.py::Class.method", "path/to/file.py::function"]}}}}
List up to 5 locations, most likely first. Use ids of the form `path::Qualname` for top-level functions, \
methods (`Class.method`) or classes. Do not list test files.
Strategy: find a relevant starting point, read it, then check the code it depends on or that depends on it \
before answering. Submit before your tool calls run out."""

USER = """Repository: {repo}
Issue:
{issue}
{hint}
You have {budget} tool calls."""


def extract_json(text: str) -> dict | None:
    """First balanced JSON object in `text` (tolerates code fences and surrounding prose)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


FINAL_TOOLS = ("final", "submit", "answer", "final_answer")


def parse_final(obj: dict | None) -> list[str] | None:
    """Accept {"final": [...]} or a call to a final-style tool; None if this is not a final answer."""
    if obj is None:
        return None
    if isinstance(obj.get("final"), list):
        return [str(x) for x in obj["final"]]
    if str(obj.get("tool", "")).lower() in FINAL_TOOLS:
        args = obj.get("args")
        if isinstance(args, list):
            return [str(x) for x in args]
        if isinstance(args, dict):
            for v in args.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
                if isinstance(v, str):
                    return [v]
        return []
    return None


def run_episode(issue: str, repo: str, tools: list[Tool], llm: Callable[[list[dict]], Reply],
                budget: int = 12, hint: str = "") -> dict:
    registry = {t.name: t for t in tools}
    tool_doc = "\n".join(f"- {t.signature}: {t.description}" for t in tools)
    messages = [{"role": "system", "content": SYSTEM.format(tools=tool_doc)},
                {"role": "user", "content": USER.format(repo=repo, issue=issue.strip()[:6000],
                                                        hint=f"\n{hint}\n" if hint else "", budget=budget)}]
    steps, final = [], None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_secs": 0.0}
    t0 = time.time()
    calls = invalid = 0
    while final is None:
        reply = llm(messages)
        usage["prompt_tokens"] += reply.prompt_tokens
        usage["completion_tokens"] += reply.completion_tokens
        usage["llm_secs"] += reply.secs
        messages.append({"role": "assistant", "content": reply.content})
        obj = extract_json(reply.content)
        answer = parse_final(obj)
        if answer is not None:
            final = answer
            steps.append({"final": final, "thought": obj.get("thought", "")})
            break
        if calls >= budget or invalid >= 3:
            messages.append({"role": "user", "content": "Budget exhausted. Reply now with your final JSON answer."})
            reply = llm(messages)
            usage["prompt_tokens"] += reply.prompt_tokens
            usage["completion_tokens"] += reply.completion_tokens
            usage["llm_secs"] += reply.secs
            final = parse_final(extract_json(reply.content)) or []
            steps.append({"final": final, "forced": True})
            break
        if obj is None or not isinstance(obj.get("tool"), str):
            invalid += 1
            obs = 'Invalid reply. Respond with one JSON object: {"thought": ..., "tool": ..., "args": {...}} or a final answer.'
        else:
            calls += 1
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            obs = call_tool(registry, obj["tool"], args)
            steps.append({"tool": obj["tool"], "args": args, "thought": obj.get("thought", ""), "obs_chars": len(obs)})
        left = budget - calls
        messages.append({"role": "user", "content": f"Observation:\n{obs}\n\n[{left} tool calls left]"})
    return {"final": final, "steps": steps, "tool_calls": calls, "invalid": invalid,
            "secs": round(time.time() - t0, 2), **usage, "messages": messages}
