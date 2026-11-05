"""The model's tools for improving itself and tuning its server. Host side, so PROTECTED.

Thin wrappers: the decisions (what may be promoted, what settings are allowed, when to undo)
live in selfimprove.py and hostcontrol.py. Only the main agent may use them, and each has a
per-run limit, so a confused model cannot hammer on them.
"""
from __future__ import annotations

from .hostcontrol import CTX_MAX, CTX_MIN, KV_CHOICES, NP_MAX, NP_MIN, Tuner
from .selfimprove import Promoter
from .tools import CallContext, Tool

MAX_PROMOTIONS_PER_RUN = 3


def _main_only(ctx: CallContext) -> str | None:
    if ctx.readonly or ctx.depth > 0:
        return "ERROR: only the main agent can use this tool."
    return None


class PromoteChangesTool(Tool):
    name = "promote_changes"
    description = ("Make your improvements to the agent's own code (the files you changed under "
                   "/workspace/agent_dev) take effect from the next start. Call it only after compileall and "
                   "the unit tests pass in agent_dev. It re-checks everything itself, applies only what is "
                   "safe, undoes it automatically if the real code then fails its tests, and tells you what "
                   "was promoted and what needs a human to approve.")

    def __init__(self, promoter: Promoter, auto: bool, max_calls: int = MAX_PROMOTIONS_PER_RUN):
        self.promoter, self.auto, self.max_calls, self.calls = promoter, auto, max_calls, 0

    def run(self, args: dict, ctx: CallContext) -> str:
        refused = _main_only(ctx)
        if refused:
            return refused
        if self.calls >= self.max_calls:
            return f"limit reached: at most {self.max_calls} promotions per run."
        self.calls += 1
        try:
            return self.promoter.promote(auto=self.auto).text()
        except FileNotFoundError as e:
            return f"ERROR: {e}"


class ServerStatusTool(Tool):
    name = "server_status"
    description = ("Show the model server's current settings (context window, parallel slots, KV cache "
                   "precision), how full the GPU memory is, and what you may change with tune_server.")

    def __init__(self, tuner: Tuner, llm):
        self.tuner, self.llm = tuner, llm

    def run(self, args: dict, ctx: CallContext) -> str:
        return _main_only(ctx) or self.tuner.status_text(self.llm.context_size())


class TuneServerTool(Tool):
    name = "tune_server"
    description = (f"Change the model server's context window (ctx: tokens per slot, {CTX_MIN}-{CTX_MAX}), "
                   f"parallel slots (np, {NP_MIN}-{NP_MAX}) or KV cache precision (kv_cache: "
                   f"{', '.join(KV_CHOICES)}; lower precision leaves more room for context). The server restarts "
                   "and is checked; if it is slow or the GPU is nearly full, the old settings are restored and "
                   "you are told. Use it when you keep running out of context ([context note] messages, or "
                   "'context full'). Check server_status first. Limited to a few changes per run.")
    parameters = {"type": "object", "properties": {
        "ctx": {"type": "integer", "description": f"context window per slot, {CTX_MIN}-{CTX_MAX}"},
        "np": {"type": "integer", "description": f"parallel slots, {NP_MIN}-{NP_MAX}"},
        "kv_cache": {"type": "string", "enum": list(KV_CHOICES)}}}

    def __init__(self, tuner: Tuner, llm):
        self.tuner, self.llm = tuner, llm

    def run(self, args: dict, ctx: CallContext) -> str:
        refused = _main_only(ctx)
        if refused:
            return refused
        numbers = {}
        for key in ("ctx", "np"):
            value = args.get(key)
            if value is None:
                continue
            try:
                numbers[key] = int(value)
            except (TypeError, ValueError):
                return f"FAILED: {key} must be a whole number."
        ok, message = self.tuner.apply(numbers.get("ctx"), numbers.get("np"), args.get("kv_cache"))
        if ok:
            self.llm.forget_window()            # the server's window may have changed: ask again
        return ("OK: " if ok else "FAILED: ") + message
