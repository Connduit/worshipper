"""A few words on what the current step is doing, for the live status line.

While the model thinks, the only thing a step knows about its own purpose is what it is reacting
to: the goal (first step) or the last action and how it went. Showing that turns
"waiting for the model... 42s" into "reacting to `npm test` (exit 1), waiting for the model... 42s".
"""
from __future__ import annotations

import json
import re

from .context import NOTE_PREFIX


def short(text, n: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= n else flat[: max(n - 3, 1)].rstrip() + "..."


def _args(raw) -> dict:
    try:
        a = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return a if isinstance(a, dict) else {}


def _summary(name: str, args: dict) -> str:
    """The call in a few words: `npm test`, write_file src/main.js, ask_friend reviewer..."""
    if name == "run_shell":
        return f"`{short(args.get('command', ''), 40)}`"
    if name == "write_file":
        return f"write_file {short(args.get('path', ''), 40)}" + (" (append)" if args.get("append") else "")
    if name in ("ask_friend", "delegate"):
        return f"{name} {short(args.get('friend', ''), 20)}".rstrip()
    return name


def _outcome(name: str, result) -> str:
    text = str(result or "")
    m = re.match(r"exit code: (-?\d+)", text)
    if name == "run_shell" and m:
        return f" (exit {m.group(1)})"
    if "timed out" in text[:80]:
        return " (timed out)"
    if text.startswith(("ERROR", "FAILED", "NOT PROMOTED")):
        return " (failed)"
    return ""


def running(name: str, args: dict) -> str:
    """What a tool that is executing right now is doing."""
    if name == "write_file":
        return f"writing {short(args.get('path', ''), 40)}"
    nice = {"promote_changes": "promoting your changes", "tune_server": "restarting the model server",
            "server_status": "checking the model server", "get_time": "checking the time"}
    return nice.get(name) or f"running {_summary(name, args)}"


def reacting_to(messages: list) -> str:
    """What the model is about to respond to, from the end of the history ('' if unknown)."""
    if not messages:
        return ""
    last = messages[-1]
    if last.get("role") == "user":
        text = str(last.get("content") or "")
        if text.startswith("GOAL: "):
            return f'starting on "{short(text[6:], 40)}"'
        if text.startswith(NOTE_PREFIX):
            return "continuing after trimming old steps"
        return "your last message"
    if last.get("role") == "tool":
        wanted = last.get("tool_call_id")
        for m in reversed(messages[:-1]):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    if tc.get("id") == wanted:
                        fn = tc.get("function") or {}
                        name = fn.get("name") or "a tool"
                        return f"reacting to {_summary(name, _args(fn.get('arguments')))}{_outcome(name, last.get('content'))}"
                break
        return "reacting to a tool result"
    return ""
