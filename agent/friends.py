"""Friends: other "minds" the main agent can collaborate with.

advisor  text-only specialist with a persistent conversation   (tool: ask_friend)
worker   a full sub-agent with its own tool loop and shell     (tool: delegate)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from .agent import FINISHED, MAX_STEPS, Agent
from .config import Config
from .llm import LLM
from .prompts import worker_system
from .storage import EventLog, FriendMemory
from .tools import CallContext, Tool, ToolPool
from .util import say, truncate

KINDS = ("advisor", "worker")
DEFAULT_WORKER_TOOLS = ("run_shell", "get_time")

# Same shape as an entry in friends.json.
DEFAULT_FRIENDS: dict[str, dict] = {
    "planner": {
        "kind": "advisor",
        "desc": "breaks a goal into small, concrete steps and spots pitfalls",
        "prompt": ("You are a planner. Break the goal you are given into small, concrete, "
                   "ordered steps. Point out missing information and likely pitfalls. "
                   "Be concise (under 200 words)."),
    },
    "reviewer": {
        "kind": "advisor",
        "desc": "critiques code, plans and results; finds bugs and risks",
        "prompt": ("You are a strict reviewer. Read what you are given and list concrete "
                   "bugs, risks and missing cases, most important first. Say clearly when "
                   "something looks fine. Be concise (under 200 words)."),
    },
    "tester": {
        "kind": "advisor",
        "desc": "designs tests and edge cases",
        "prompt": ("You design tests. Given a change or feature, list the most valuable "
                   "test cases and edge cases, with the exact commands or inputs to try "
                   "and the expected result. Be concise (under 200 words)."),
    },
    "coder": {
        "kind": "worker",
        "desc": "implements a well-specified coding task in /workspace and reports back",
        "prompt": ("You are a careful programmer. Make small, targeted changes and verify "
                   "each one by running it."),
        "tools": list(DEFAULT_WORKER_TOOLS),
        "max_steps": 15,
    },
    "inspector": {
        "kind": "worker",
        "desc": "read-only investigator: reads files, runs checks, reports findings",
        "prompt": "You investigate and report facts. You never modify anything.",
        "tools": ["run_shell"],
        "max_steps": 10,
        "readonly": True,
    },
}


# --------------------------------------------------------------------------- specs
@dataclass(frozen=True)
class FriendSpec:
    name: str
    kind: str                                   # "advisor" | "worker"
    prompt: str
    desc: str = ""
    tools: tuple[str, ...] = DEFAULT_WORKER_TOOLS   # workers only
    max_steps: int = 12                             # workers only
    readonly: bool = False                          # workers only
    url: str | None = None                          # own llama-server, optional
    model: str | None = None

    @classmethod
    def from_dict(cls, name: str, d) -> "FriendSpec":
        """Validate one friends.json entry; raises ValueError if it is unusable."""
        if not isinstance(d, dict) or d.get("kind") not in KINDS or not d.get("prompt"):
            raise ValueError("needs kind advisor|worker and a prompt")
        tools = d.get("tools", DEFAULT_WORKER_TOOLS)
        if not isinstance(tools, (list, tuple)) or not all(isinstance(t, str) for t in tools):
            raise ValueError("tools must be a list of tool names")
        try:
            max_steps = int(d.get("max_steps", 12))
        except (TypeError, ValueError):
            raise ValueError("max_steps must be an integer") from None
        return cls(name=name, kind=d["kind"], prompt=str(d["prompt"]), desc=str(d.get("desc", "")),
                   tools=tuple(tools), max_steps=max_steps, readonly=bool(d.get("readonly", False)),
                   url=d.get("url") or None, model=d.get("model") or None)


class FriendRegistry:
    """The set of configured friends: built-ins, plus/overridden by friends.json."""

    def __init__(self, specs: dict[str, FriendSpec]):
        self._specs = specs

    @classmethod
    def load(cls, path: str, enabled: bool = True) -> "FriendRegistry":
        """Built-in friends, replaced/extended by `path` (a null entry removes a friend)."""
        if not enabled:
            return cls({})
        raw: dict = {k: dict(v) for k, v in DEFAULT_FRIENDS.items()}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    extra = json.load(f)
                if not isinstance(extra, dict):
                    raise ValueError("top level must be an object")
                for name, entry in extra.items():
                    if entry is None:
                        raw.pop(name, None)
                    else:
                        raw[name] = entry
            except (OSError, ValueError) as e:      # JSONDecodeError is a ValueError
                print(f"[warn] could not read {path}: {e}")
        specs = {}
        for name, entry in raw.items():
            try:
                specs[name] = FriendSpec.from_dict(name, entry)
            except ValueError as e:
                print(f"[warn] ignoring friend {name!r}: {e}")
        return cls(specs)

    def __bool__(self) -> bool:
        return bool(self._specs)

    def __iter__(self):
        return iter(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def advisors(self) -> list[FriendSpec]:
        return [f for f in self._specs.values() if f.kind == "advisor"]

    def workers(self) -> list[FriendSpec]:
        return [f for f in self._specs.values() if f.kind == "worker"]

    def get(self, name, kind: str) -> FriendSpec | None:
        spec = self._specs.get(name)
        return spec if spec and spec.kind == kind else None

    def describe(self, *, config_file: str, max_depth: int, default_url: str,
                 default_model: str) -> str:
        """Human-readable listing for --friends."""
        if not self:
            return "No friends configured."
        src = config_file if os.path.exists(config_file) else "none found"
        lines = [f"Friends (config file: {src}; MAX_DEPTH={max_depth})", ""]
        for f in self:
            lines.append(f"{f.name:<12} {f.kind:<8} {f.desc}")
            if f.kind == "worker":
                bits = [f"tools={list(f.tools)}", f"max_steps={f.max_steps}"]
                if f.readonly:
                    bits.append("readonly")
                lines.append(" " * 21 + ", ".join(bits))
            if f.url or f.model:
                lines.append(" " * 21 + f"url={f.url or default_url} model={f.model or default_model}")
        if max_depth < 1 and self.workers():
            lines += ["", "Note: MAX_DEPTH=0, so workers cannot be delegated to."]
        return "\n".join(lines)


# --------------------------------------------------------------------------- service
class FriendService:
    """Does the actual asking and delegating."""

    def __init__(self, cfg: Config, registry: FriendRegistry, default_llm: LLM,
                 memory: FriendMemory, log: EventLog,
                 llm_factory: Callable[[str, str], LLM] | None = None):
        self.cfg = cfg
        self.registry = registry
        self.default_llm = default_llm
        self.memory = memory
        self.log = log
        self._make_llm = llm_factory or (lambda url, model: LLM(url, model, log))
        self._llms: dict[tuple[str, str], LLM] = {}
        self.tool_pool: ToolPool | None = None      # set by App; workers pick their tools from it

    def _llm_for(self, spec: FriendSpec) -> LLM:
        """The shared LLM, or the friend's own server/model if it configured one."""
        if not spec.url and not spec.model:
            return self.default_llm
        key = (spec.url or self.default_llm.base_url, spec.model or self.default_llm.model)
        if key not in self._llms:
            self._llms[key] = self._make_llm(*key)
        return self._llms[key]

    def ask(self, name, message, depth: int) -> str:
        """Advisor: one chat turn with a persistent, text-only specialist."""
        spec = self.registry.get(name, "advisor")
        if spec is None:
            workers = [f.name for f in self.registry.workers()]
            hint = f" (workers are used with delegate: {workers})" if workers else ""
            return (f"ERROR: unknown advisor {name!r}. Advisors: "
                    f"{[f.name for f in self.registry.advisors()]}{hint}")
        if not message:
            return "ERROR: missing argument 'message'"
        message = str(message)

        history = self.memory.prepare(spec.name, spec.prompt, message)
        say(depth, f"[ask {spec.name}] {truncate(message, 300)}")
        resp = self._llm_for(spec).chat(history, depth=depth, label=spec.name)
        if resp is None:
            self.memory.rollback(spec.name)
            return f"ERROR: {spec.name} could not be reached (the model server kept failing)"
        reply = (resp.choices[0].message.content or "").strip() or "(no reply)"
        self.memory.commit(spec.name, reply)
        self.log.log("ask_friend", friend=spec.name, depth=depth, message=message, reply=reply)
        say(depth, f"[{spec.name} replies] {truncate(reply, 500)}")
        return truncate(reply)

    def delegate(self, name, task, depth: int) -> str:
        """Worker: run a sub-agent for one task and return its report."""
        spec = self.registry.get(name, "worker")
        if spec is None:
            advisors = [f.name for f in self.registry.advisors()]
            hint = f" (advisors are used with ask_friend: {advisors})" if advisors else ""
            return (f"ERROR: unknown worker {name!r}. Workers: "
                    f"{[f.name for f in self.registry.workers()]}{hint}")
        if not task:
            return "ERROR: missing argument 'task'"
        if depth >= self.cfg.max_depth:
            return "ERROR: delegation depth limit reached; do this task yourself"
        if self.tool_pool is None:
            raise RuntimeError("FriendService.tool_pool was not set")

        agent = Agent(spec.name, self._llm_for(spec), self.tool_pool.select(spec.tools, depth + 1),
                      spec.max_steps, self.log, depth=depth + 1, readonly=spec.readonly)
        messages = [{"role": "system", "content": worker_system(spec)},
                    {"role": "user", "content": f"TASK: {task}"}]
        say(depth, f"[delegate -> {spec.name}] {truncate(str(task), 300)}")
        self.log.log("delegate", friend=spec.name, depth=depth, task=task)

        result = agent.run(messages)
        if result.status == FINISHED:
            return truncate(f"[{spec.name} finished]\n{result.text}")
        if result.status == MAX_STEPS:
            return truncate(f"[{spec.name} ran out of steps without finishing]\n"
                            f"Last message: {result.text or '(none)'}")
        return f"[{spec.name} failed: the model server kept failing]"


# --------------------------------------------------------------------------- tools
class AskFriendTool(Tool):
    name = "ask_friend"
    description = ("Ask an advisor friend for advice, review or ideas. Text only; the "
                   "friend cannot see your conversation, so include the context.")

    def __init__(self, service: FriendService):
        self.service = service
        self.parameters = {"type": "object",
                           "properties": {"friend": {"type": "string",
                                                     "enum": [f.name for f in service.registry.advisors()]},
                                          "message": {"type": "string"}},
                           "required": ["friend", "message"]}

    def run(self, args: dict, ctx: CallContext) -> str:
        return self.service.ask(args.get("friend"), args.get("message"), ctx.depth)


class DelegateTool(Tool):
    name = "delegate"
    description = ("Hand a self-contained task to a worker friend, which runs its own agent loop "
                   "in the sandbox and returns a report. It starts fresh, so describe the task fully.")
    delegating = True

    def __init__(self, service: FriendService):
        self.service = service
        self.parameters = {"type": "object",
                           "properties": {"friend": {"type": "string",
                                                     "enum": [f.name for f in service.registry.workers()]},
                                          "task": {"type": "string"}},
                           "required": ["friend", "task"]}

    def run(self, args: dict, ctx: CallContext) -> str:
        return self.service.delegate(args.get("friend"), args.get("task"), ctx.depth)
