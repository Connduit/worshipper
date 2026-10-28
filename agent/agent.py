"""The agent loop: ask the model, run the tool it picked, show it the result, repeat."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .console import Console
from .context import compact, total_chars
from .llm import LLM, ContextFull
from .storage import EventLog
from .tools import CallContext, Tool
from .util import truncate

FINISHED, MAX_STEPS, LLM_FAILED, CONTEXT_FULL = "finished", "max_steps", "llm_failed", "context_full"

# Context management (see context.py). When the history is estimated to come within RESERVE tokens
# of the window, shrink it to TARGET_PROACTIVE of the window. If the server says it is full anyway,
# shrink it harder, to TARGET_REACTIVE, and try once more.
TARGET_PROACTIVE, TARGET_REACTIVE = 0.50, 0.40
MAX_RESERVE = 4096              # tokens kept free for the model's reply

NUDGE = "You must call a tool. Use one of your tools to act, or finish if done."


@dataclass
class RunResult:
    status: str         # FINISHED | MAX_STEPS | LLM_FAILED | CONTEXT_FULL
    text: str = ""      # the finish answer, or the last thing the model said


class Agent:
    """Runs one conversation. Used for the main agent (depth 0) and for delegated workers."""

    def __init__(self, name: str, llm: LLM, tools: list[Tool], max_steps: int, log: EventLog,
                 console: Console, *, depth: int = 0, readonly: bool = False,
                 on_save: Callable[[], None] | None = None,
                 on_drop: Callable[[list[dict]], None] | None = None):
        self.name = name
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.log = log
        self.console = console
        self.depth = depth
        self.readonly = readonly
        self.on_save = on_save or (lambda: None)
        self.on_drop = on_drop
        self._tokens_per_char = 0.33        # refined from the server's own token counts

    @property
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, messages: list) -> RunResult:
        """Run the tool loop on `messages` (mutated in place, so history carries over)."""
        d = self.depth
        tag = "" if d == 0 else f"{self.name} "

        for step in range(1, self.max_steps + 1):
            self.console.debug(d, f"\n===== {tag}step {step} =====")
            resp = self._ask(messages, step)
            if isinstance(resp, RunResult):         # could not get an answer: say why
                self.console.debug(d, f"[STOPPED] {resp.text}; see {self.log.path}")
                self.on_save()
                return resp
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if msg.content:
                self.console.debug(d, f"[reason] {msg.content.strip()}")

            # Small local models sometimes forget to call a tool; nudge them.
            if not msg.tool_calls:
                messages.append({"role": "user", "content": NUDGE})
                continue

            finished, answer = False, ""
            for call in msg.tool_calls:
                observation, done, call_answer = self._handle(call, step)
                if done:
                    finished, answer = True, call_answer
                self.console.debug(d, f"[observe] {truncate(observation, 500)}")
                # Every tool call must get an answer, or the history is invalid next turn.
                messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

            self.on_save()
            if finished:
                return RunResult(FINISHED, answer)

        self.console.debug(d, f"\n[STOPPED] hit max steps ({self.max_steps})")
        self.log.log("max_steps", agent=self.name, depth=d)
        self.on_save()
        last = next((m.get("content") for m in reversed(messages)
                     if m.get("role") == "assistant" and m.get("content")), "")
        return RunResult(MAX_STEPS, last or "")

    # -- talking to the model, with context management ---------------------------------------
    def _schema_chars(self) -> int:
        return len(json.dumps(self.schemas))

    def _ask(self, messages: list, step: int):
        """One model call. Makes room first when the history is close to the window, and shrinks
        it and retries once if the server says it does not fit. Returns the response, or a
        RunResult when the run cannot continue."""
        for attempt in (1, 2):
            self._make_room(messages)
            chars = total_chars(messages, self._schema_chars())
            try:
                resp = self.llm.chat(messages, self.schemas, depth=self.depth, label=self.name,
                                     step=step, raise_context_full=True)
            except ContextFull as e:
                n_ctx = e.n_ctx or self.llm.context_size()
                if attempt == 2 or not n_ctx or not self._compact(messages, int(n_ctx * TARGET_REACTIVE),
                                                                  n_ctx, "the server said it was full"):
                    return RunResult(CONTEXT_FULL, self.llm.failure_text())
                continue
            if resp is None:
                return RunResult(LLM_FAILED, self.llm.failure_text())
            self._calibrate(resp, chars)
            return resp
        return RunResult(CONTEXT_FULL, self.llm.failure_text())      # not reached

    def _calibrate(self, resp, sent_chars: int) -> None:
        """Learn how many tokens a character costs for this model and this kind of text."""
        prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        if isinstance(prompt_tokens, int) and prompt_tokens > 0 and sent_chars > 0:
            self._tokens_per_char = min(1.0, max(0.15, prompt_tokens / sent_chars))

    def _make_room(self, messages: list) -> None:
        n_ctx = self.llm.context_size()
        if not n_ctx:
            return
        estimate = total_chars(messages, self._schema_chars()) * self._tokens_per_char
        if estimate > n_ctx - min(MAX_RESERVE, n_ctx // 4):
            self._compact(messages, int(n_ctx * TARGET_PROACTIVE), n_ctx, "nearly full")

    def _compact(self, messages: list, target_tokens: int, n_ctx: int, why: str) -> bool:
        """Shrink the history to about target_tokens. Returns whether anything was shrunk."""
        extra = self._schema_chars()
        report = compact(messages, int(target_tokens / self._tokens_per_char), extra_chars=extra, on_drop=self.on_drop)
        if not report.changed:
            return False
        before, after = (int(c * self._tokens_per_char) for c in (report.before, report.after))
        what = ", ".join(x for x in (
            f"trimmed {report.trimmed_outputs} old tool outputs" if report.trimmed_outputs else "",
            f"trimmed {report.trimmed_calls} long tool calls" if report.trimmed_calls else "",
            f"dropped {report.dropped_steps} old steps" if report.dropped_steps else "") if x) or "dropped old reasoning"
        self.console.debug(self.depth, f"[context] {why}: ~{before} -> ~{after} of {n_ctx} tokens ({what})")
        self.log.log("compact", agent=self.name, reason=why, before_tokens=before, after_tokens=after,
                     n_ctx=n_ctx, trimmed_outputs=report.trimmed_outputs, trimmed_calls=report.trimmed_calls,
                     dropped_steps=report.dropped_steps)
        self.on_save()
        return True

    def _handle(self, call, step: int) -> tuple[str, bool, str]:
        """Run one tool call. Returns (observation, finished?, the finish answer)."""
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = None
        if not isinstance(args, dict):
            args = None

        done, answer = False, ""
        tool = self.tools.get(name)
        if args is None:
            observation = "ERROR: tool arguments were not valid JSON. Try again."
        elif tool is None:
            observation = f"ERROR: tool {name} is not available to you. Available: {sorted(self.tools)}"
        else:
            if tool.is_final:
                answer = str(args.get("answer") or args.get("summary") or "").strip()   # old name still accepted
                if answer:
                    self.console.debug(self.depth, f"\n[DONE] {answer}")
                    self.log.log("finish", agent=self.name, depth=self.depth, step=step, answer=answer)
                    done = True
                    observation = self._run_tool(tool, args)
                else:       # an empty finish would leave the user with nothing; make the model retry
                    observation = ("ERROR: finish needs a non-empty `answer`. Put the actual answer "
                                   "or result in it, not a description of what you did.")
            else:
                self.console.debug(self.depth, f"[act] {name}({args})")
                observation = self._run_tool(tool, args)

        self.log.log("step", agent=self.name, depth=self.depth, step=step,
                     tool=name, args=args, observation=observation)
        return observation, done, answer

    def _run_tool(self, tool: Tool, args: dict) -> str:
        try:
            return tool.run(args, CallContext(self.depth, self.readonly, self.name))
        except KeyError as e:
            return f"ERROR: missing argument {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"
