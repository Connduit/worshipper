"""The tools an agent can call.

A Tool knows its own JSON schema and how to run itself. A ToolPool holds all
tools; each agent gets a subset of it (see ToolPool.select).
"""
from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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


class WriteFileTool(Tool):
    """Create or extend a text file in the workspace.

    Far cheaper in tokens than a heredoc inside run_shell (whose text is escaped inside JSON),
    and a big file can be written in pieces. The write happens on the host side, so the path
    is checked: it must stay inside the workspace, also after following symlinks (a command in
    the sandbox could have made one that points elsewhere).
    """
    name = "write_file"
    MAX_BYTES = 200_000
    description = ("Create or overwrite a text file in /workspace. Prefer this to heredocs in run_shell: it "
                   "needs far fewer tokens. For a big file write it in pieces of about 100 lines: the first "
                   "call creates it, the following calls pass append=true.")
    parameters = {"type": "object",
                  "properties": {"path": {"type": "string",
                                          "description": "Relative to /workspace, e.g. src/main.js"},
                                 "content": {"type": "string"},
                                 "append": {"type": "boolean",
                                            "description": "Add to the end of the file instead of replacing it."}},
                  "required": ["path", "content"]}

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace)

    def run(self, args: dict, ctx: CallContext) -> str:
        if ctx.readonly:
            return "ERROR: you are a read-only worker and cannot write files."
        content = args["content"]
        if not isinstance(content, str):
            return "ERROR: `content` must be a string."
        base = self.workspace.resolve()
        rel = str(args["path"]).strip()
        if rel == "/workspace" or rel.startswith("/workspace/"):
            rel = rel[len("/workspace"):].lstrip("/")
        if not rel or rel in (".", "./"):
            return "ERROR: give the path of a file, like src/main.js."
        if rel.startswith("/"):
            return "ERROR: use a path inside /workspace (relative, like src/main.js), not an absolute path."
        target = (base / rel).resolve()                     # follows symlinks
        if base not in target.parents:
            return "ERROR: that path is outside /workspace."
        if target.is_dir():
            return f"ERROR: {target.relative_to(base)} is a directory."
        data = content.encode("utf-8")
        if len(data) > self.MAX_BYTES:
            return (f"ERROR: {len(data)} bytes in one call is too much (limit {self.MAX_BYTES}). Write it in "
                    "pieces: the first call creates the file, later calls use append=true.")
        append = bool(args.get("append"))
        target.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(target, flags, 0o644), "wb") as f:
            f.write(data)
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return (f"{'appended' if append else 'wrote'} {len(data)} bytes ({lines} lines) to "
                f"{target.relative_to(base)}; the file is now {target.stat().st_size} bytes")


class FinishTool(Tool):
    name = "finish"
    description = ("Call when the goal is done. `answer` is shown to the user as the final result, "
                   "so put the actual answer or result in it: the content itself, not a "
                   "description of what you did.")
    parameters = {"type": "object",
                  "properties": {"answer": {"type": "string",
                                            "description": "The complete final answer or result."}},
                  "required": ["answer"]}
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
