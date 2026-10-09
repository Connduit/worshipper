"""The agent loop: ask the model, run the tool it picked, show it the result, repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .llm import LLM
from .storage import EventLog
from .tools import CallContext, Tool
from .util import say, truncate

FINISHED, MAX_STEPS, LLM_FAILED = "finished", "max_steps", "llm_failed"

NUDGE = "You must call a tool. Use one of your tools to act, or finish if done."


@dataclass
class RunResult:
    status: str         # FINISHED | MAX_STEPS | LLM_FAILED
    text: str = ""      # finish summary, or the last thing the model said


class Agent:
    """Runs one conversation. Used for the main agent (depth 0) and for delegated workers."""

    def __init__(self, name: str, llm: LLM, tools: list[Tool], max_steps: int, log: EventLog,
                 *, depth: int = 0, readonly: bool = False,
                 on_save: Callable[[], None] | None = None):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.log = log
        self.depth = depth
        self.readonly = readonly
        self.on_save = on_save or (lambda: None)

    @property
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, messages: list) -> RunResult:
        """Run the tool loop on `messages` (mutated in place, so history carries over)."""
        d = self.depth
        tag = "" if d == 0 else f"{self.name} "

        for step in range(1, self.max_steps + 1):
            say(d, f"\n===== {tag}step {step} =====")
            resp = self.llm.chat(messages, self.schemas, depth=d, label=self.name, step=step)
            if resp is None:
                say(d, f"[STOPPED] the model server kept failing; see {self.log.path}")
                self.on_save()
                return RunResult(LLM_FAILED)
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.content:
                say(d, f"[reason] {msg.content.strip()}")

            # Small local models sometimes forget to call a tool; nudge them.
            if not msg.tool_calls:
                messages.append({"role": "user", "content": NUDGE})
                continue

            finished, summary = False, ""
            for call in msg.tool_calls:
                observation, done, call_summary = self._handle(call, step)
                if done:
                    finished, summary = True, call_summary
                say(d, f"[observe] {truncate(observation, 500)}")
                # Every tool call must get an answer, or the history is invalid next turn.
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

            self.on_save()
            if finished:
                return RunResult(FINISHED, summary)

        say(d, f"\n[STOPPED] hit max steps ({self.max_steps})")
        self.log.log("max_steps", agent=self.name, depth=d)
        self.on_save()
        last = next((m.get("content") for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")), "")
        return RunResult(MAX_STEPS, last or "")

    def _handle(self, call, step: int) -> tuple[str, bool, str]:
        """Run one tool call. Returns (observation, finished?, finish summary)."""
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            args = None

        done, summary = False, ""
        tool = self.tools.get(name)
        if args is None:
            observation = "ERROR: tool arguments were not valid JSON. Try again."
        elif tool is None:
            observation = f"ERROR: tool {name} is not available to you. Available: {sorted(self.tools)}"
        else:
            if tool.is_final:
                summary = args.get("summary") or ""
                say(self.depth, f"\n[DONE] {summary}")
                self.log.log("finish", agent=self.name, depth=self.depth, step=step, summary=summary)
                done = True
            else:
                say(self.depth, f"[act] {name}({args})")
            observation = self._run_tool(tool, args)

        self.log.log("step", agent=self.name, depth=self.depth, step=step,
                     tool=name, args=args, observation=observation)
        return observation, done, summary

    def _run_tool(self, tool: Tool, args: dict) -> str:
        try:
            return tool.run(args, CallContext(self.depth, self.readonly, self.name))
        except KeyError as e:
            return f"ERROR: missing argument {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
