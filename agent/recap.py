"""A short reminder of what a saved conversation was about, shown when it is resumed.

The history already holds everything: each goal is a user message starting with "GOAL: ", and
the agent's answer is the text it passed to `finish`. Nothing extra is stored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .context import NOTE_PREFIX
from .util import fmt_duration

GOAL_PREFIX = "GOAL: "


@dataclass
class Exchange:
    goal: str
    answer: str | None = None       # what the agent gave to `finish`; None if the run never got there


def exchanges(messages: list) -> list[Exchange]:
    """The goals in the history, oldest first, each with the agent's final answer if it has one."""
    out: list[Exchange] = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "user" and isinstance(content, str) and content.startswith(GOAL_PREFIX):
            out.append(Exchange(content[len(GOAL_PREFIX):].strip()))
        elif out and m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") != "finish":
                    continue
                try:
                    answer = json.loads(fn.get("arguments") or "{}").get("answer")
                except (ValueError, AttributeError):
                    answer = None
                if isinstance(answer, str) and answer.strip():
                    out[-1].answer = answer.strip()
    return out


def _one_line(text: str, width: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: max(width - 3, 1)].rstrip() + "..."


def format_recap(messages: list, age_seconds: float | None = None, *, max_goals: int = 3,
                 width: int = 96) -> str:
    """The last few goals and answers of a resumed conversation, or "" if there are none."""
    done = exchanges(messages)
    if not done:
        return ""
    shown = done[-max_goals:]
    facts = ([f"last used {fmt_duration(age_seconds)} ago"] if age_seconds is not None and age_seconds >= 0 else [])
    facts.append(f"{len(done)} goal{'s' if len(done) != 1 else ''} so far")
    lines = [f"Resuming your previous conversation ({', '.join(facts)}). "
             + ("The last one:" if len(shown) == 1 else f"The last {len(shown)}:")]
    first = len(done) - len(shown) + 1
    for n, e in enumerate(shown, start=first):
        lines.append(f"  {n}. you:   {_one_line(e.goal, width)}")
        lines.append("     agent: " + (_one_line(e.answer, width) if e.answer else
                                       "(no final answer: that run was interrupted or stopped early)"))
    if any(m.get("role") == "user" and str(m.get("content", "")).startswith(NOTE_PREFIX) for m in messages):
        lines.append("  (older steps were trimmed to save context; the files in /workspace are unchanged)")
    return "\n".join(lines)
