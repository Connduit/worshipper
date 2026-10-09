"""The tools an agent can call.

A Tool knows its own JSON schema and how to run itself. A ToolPool holds all
tools; each agent gets a subset of it (see ToolPool.select).
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .sandbox import Sandbox
from .util import truncate


@dataclass(frozen=True)
class CallContext:
    """Who is calling a tool."""
    depth: int = 0              # 0 = main agent, 1 = a delegated worker, ...
    readonly: bool = False      # the caller is a read-only worker
    name: str = "main"


class Tool(ABC):
    name: str
    description: str
    parameters: dict = {"type": "object", "properties": {}}
    is_final = False            # calling it ends the agent's run (finish)
    delegating = False          # it starts nested agents; hidden once max depth is reached

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    @abstractmethod
    def run(self, args: dict, ctx: CallContext) -> str:
        """Return the observation the model will see. May raise KeyError for a missing argument."""


class GetTimeTool(Tool):
    name = "get_time"
    description = "Return the current date and time."

    def run(self, args: dict, ctx: CallContext) -> str:
        return datetime.now().isoformat()


class RunShellTool(Tool):
    name = "run_shell"

    def __init__(self, sandbox: Sandbox, default_timeout: int, max_timeout: int):
        self.sandbox = sandbox
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.description = (f"Run a bash command inside the sandbox. Default timeout {default_timeout}s; "
                            f"set `timeout` (seconds, max {max_timeout}) for slow commands.")
        self.parameters = {"type": "object",
                           "properties": {"command": {"type": "string"},
                                          "timeout": {"type": "integer"}},
                           "required": ["command"]}

    def run(self, args: dict, ctx: CallContext) -> str:
        try:
            timeout = int(args.get("timeout") or self.default_timeout)
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(1, min(timeout, self.max_timeout))
        try:
            r = self.sandbox.run(args["command"], timeout, ctx.readonly)
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout}s"
        return truncate(f"exit code: {r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")


class FinishTool(Tool):
    name = "finish"
    description = "Call when the goal is complete."
    parameters = {"type": "object",
                  "properties": {"summary": {"type": "string"}},
                  "required": ["summary"]}
    is_final = True

    def run(self, args: dict, ctx: CallContext) -> str:
        return "Goal marked complete. Wait for the next goal."


class ToolPool:
    """All available tools by name. Must contain a FinishTool."""

    def __init__(self, tools: Iterable[Tool], max_depth: int):
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self.max_depth = max_depth

    def select(self, names: Iterable[str], depth: int) -> list[Tool]:
        """Tools for an agent at `depth`; `finish` is always included, unknown names are skipped."""
        out = []
        for n in names:
            tool = self._tools.get(n)
            if tool is None or tool.is_final:
                continue
            if tool.delegating and depth >= self.max_depth:
                continue
            out.append(tool)
        out.append(self._tools["finish"])
        return out
