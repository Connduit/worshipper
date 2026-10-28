"""Keeping the conversation inside the model's context window.

Every step adds to the history, and a local model has a fixed window (the `CTX` you gave
llama-server). When the history no longer fits, the server either refuses the request or cuts
the model's reply off mid-way. `compact` prevents that by shrinking the *old, bulky* parts of
the history, cheapest first, and only as much as needed:

  1. reasoning text of old assistant turns is dropped
  2. old tool outputs are cut to their beginning and end
  3. long arguments of old tool calls (e.g. a whole file that was written) are cut the same way
  4. whole old steps are dropped, oldest first, and one note says that this happened

It never touches the system prompt, the current goal, or the last few messages (as a last resort,
when that is still not enough, only the newest step), and it keeps every tool call together with
its results (a result without its call is invalid). No model
call is needed, so it is instant and deterministic. Files in the workspace are untouched: the
model can always look at them again.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

KEEP_RECENT = 6                 # the newest messages are never touched
LAST_RESORT_KEEP = 2            # ...except, if nothing else is enough, all but the newest step
SNIP_HEAD, SNIP_TAIL = 200, 100
NOTE_PREFIX = "[context note]"
NOTE = (f"{NOTE_PREFIX} Earlier steps of this conversation were removed to save space. Files in "
        "/workspace are unchanged: use ls and cat to re-check what exists before continuing.")


def message_chars(m: dict) -> int:
    """Rough size of one message in characters (content plus tool-call arguments)."""
    n = 16 + len(m.get("reasoning_content") or "")      # sent back with the history, so it costs tokens
    content = m.get("content")
    if isinstance(content, str):
        n += len(content)
    elif content:
        n += len(json.dumps(content))
    for tc in m.get("tool_calls") or []:
        f = tc.get("function") or {}
        n += len(f.get("name") or "") + len(f.get("arguments") or "") + 24
    return n


def total_chars(messages: list, extra: int = 0) -> int:
    """Size of the whole history; `extra` is for what is sent besides it (the tool schemas)."""
    return sum(message_chars(m) for m in messages) + extra


@dataclass
class Report:
    before: int                 # characters, before and after
    after: int = 0
    trimmed_outputs: int = 0
    trimmed_calls: int = 0
    dropped_steps: int = 0

    @property
    def changed(self) -> bool:
        return self.after < self.before


def _snip(text: str, head: int = SNIP_HEAD, tail: int = SNIP_TAIL) -> str:
    if len(text) <= head + tail + 80:
        return text
    cut = len(text) - head - tail
    return f"{text[:head]}\n...[{cut} chars trimmed to save context]...\n{text[-tail:]}"


def _shrink_arguments(arguments: str) -> str:
    """Shorten long string arguments of an old tool call, keeping the JSON valid (the server
    parses it again every time the history is sent)."""
    try:
        args = json.loads(arguments)
    except (TypeError, ValueError):
        args = None
    if not isinstance(args, dict):
        return json.dumps({"trimmed": "arguments too long to keep"}) if len(arguments or "") > 600 else arguments
    return json.dumps({k: _snip(v, 150, 80) if isinstance(v, str) and len(v) > 400 else v
                       for k, v in args.items()})


def _units(messages: list, start: int) -> list[tuple[int, int]]:
    """Split messages[start:] into units that must stay together: a plain message, or an
    assistant message with tool calls plus the tool results that answer them."""
    units, i = [], start
    while i < len(messages):
        j = i + 1
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
        units.append((i, j))
        i = j
    return units


def compact(messages: list, target_chars: int, *, extra_chars: int = 0,
            keep_recent: int = KEEP_RECENT,
            on_drop: Callable[[list[dict]], None] | None = None) -> Report:
    """Shrink `messages` in place until total_chars(messages, extra_chars) <= target_chars,
    or until there is nothing left that may be shrunk."""
    start = 0
    while start < len(messages) and messages[start].get("role") == "system":
        start += 1
    report = Report(before=total_chars(messages, extra_chars))

    def size() -> int:
        return total_chars(messages, extra_chars)

    def done() -> Report:
        report.after = size()
        return report

    units = _units(messages, start)
    tail_start = len(messages)                       # first protected message
    for s, e in units:
        if e > len(messages) - keep_recent:
            tail_start = s
            break

    # 1-2: old assistant reasoning and old tool outputs
    for i in range(start, tail_start):
        if messages[i].get("role") == "assistant":
            messages[i].pop("reasoning_content", None)
    if size() <= target_chars:
        return done()
    for i in range(start, tail_start):
        m = messages[i]
        if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 500:
            m["content"] = _snip(m["content"])
            report.trimmed_outputs += 1
    if size() <= target_chars:
        return done()

    # 3: long tool-call arguments and long text of old assistant turns
    for i in range(start, tail_start):
        m = messages[i]
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            new = _shrink_arguments(f.get("arguments") or "")
            if new != f.get("arguments"):
                f["arguments"] = new
                report.trimmed_calls += 1
        if isinstance(m.get("content"), str) and len(m["content"]) > 800:
            m["content"] = _snip(m["content"], 400, 200)
    if size() <= target_chars:
        return done()

    # 4: drop whole old steps, oldest first (never the current goal)
    goal = max((i for i, m in enumerate(messages)
                if m.get("role") == "user" and str(m.get("content", "")).startswith("GOAL:")), default=None)
    droppable = [(s, e) for s, e in units
                 if e <= tail_start and not (goal is not None and s <= goal < e)]
    note_chars = message_chars({"role": "user", "content": NOTE})
    dropped: list[tuple[int, int]] = []
    freed = 0
    for s, e in droppable:
        if size() - freed + note_chars <= target_chars:
            break
        dropped.append((s, e))
        freed += sum(message_chars(m) for m in messages[s:e])
    if dropped:
        if on_drop is not None:
            on_drop([dict(m) for s, e in dropped for m in messages[s:e]])
        gone = {i for s, e in dropped for i in range(s, e)}
        is_note = lambda m: m.get("role") == "user" and str(m.get("content", "")).startswith(NOTE_PREFIX)
        old_notes = [i for i, m in enumerate(messages) if is_note(m)]
        at = min([min(gone)] + old_notes)               # one note, where the removed material was
        rebuilt = []
        for i, m in enumerate(messages):
            if i == at:
                rebuilt.append({"role": "user", "content": NOTE})
            if i in gone or is_note(m):
                continue
            rebuilt.append(m)
        messages[:] = rebuilt
        report.dropped_steps = len(dropped)
    if size() > target_chars and keep_recent > LAST_RESORT_KEEP:
        # Still too big: the "recent" messages themselves are what is bulky (a small window).
        # Protect only the newest step this time.
        again = compact(messages, target_chars, extra_chars=extra_chars, keep_recent=LAST_RESORT_KEEP, on_drop=on_drop)
        report.trimmed_outputs += again.trimmed_outputs
        report.trimmed_calls += again.trimmed_calls
        report.dropped_steps += again.dropped_steps
    return done()
