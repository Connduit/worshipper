"""App: builds all the pieces and runs goals. The only place that knows about everything."""
from __future__ import annotations

import os
import shutil
import time

from .agent import CONTEXT_FULL, FINISHED, MAX_STEPS, Agent, RunResult
from .config import ROOT, Config
from .console import Console
from .devcopy import DevCopy
from .friends import AskFriendTool, DelegateTool, FriendRegistry, FriendService
from .inbox import TIP, Inbox
from .llm import LLM
from .hostcontrol import Tuner
from .prompts import build_system, can_self_improve, write_file_available
from .recap import format_recap
from .sandbox import make_sandbox
from .selfimprove import Promoter
from .selftools import PromoteChangesTool, ServerStatusTool, TuneServerTool
from .storage import EventLog, FriendMemory, ProjectMemory, SessionStore
from .tools import FinishTool, GetTimeTool, RunShellTool, ToolPool, WriteFileTool

MAIN_TOOLS = ("get_time", "run_shell", "write_file", "promote_changes", "server_status", "tune_server",
              "ask_friend", "delegate")


class App:
    def __init__(self, cfg: Config, llm: LLM | None = None):
        cfg.workspace.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.console = Console(cfg.debug)
        self.log = EventLog(cfg.log_file)
        self.sessions = SessionStore(cfg.session_file)
        self.project_memory = ProjectMemory(str(cfg.session_file))
        self.memory = FriendMemory(cfg.friend_memory_file, cfg.max_friend_history)
        self.devcopy = DevCopy(ROOT, cfg.dev_dir)
        self.sandbox = make_sandbox(cfg)
        self.llm = llm or LLM(cfg.llm_url, cfg.model, self.log, self.console,
                              n_ctx=cfg.context_tokens or None)
        self.registry = FriendRegistry.load(cfg.friends_file, enabled=cfg.use_friends)
        self.system = build_system(cfg, self.registry)
        self._recap = ""            # reminder of a resumed conversation, shown once (see setup)

        tools = [GetTimeTool(), RunShellTool(self.sandbox, cfg.default_timeout, cfg.max_timeout),
                 FinishTool()]
        if write_file_available(cfg):
            tools.append(WriteFileTool(cfg.workspace))
        if can_self_improve(cfg) and cfg.self_improve != "off":
            # new code is first run in the jail: the dev copy as the model left it, then the real tree
            # (shown read-only) after the files were copied
            runner = lambda command, timeout, binds=(): self.sandbox.run(command, timeout, False, binds)  # noqa: E731
            tools.append(PromoteChangesTool(Promoter(ROOT, cfg.dev_dir, runner), auto=cfg.self_improve == "auto"))
        if cfg.tune_server and (ROOT / "model.sh").exists():
            tuner = Tuner(ROOT)
            tools += [ServerStatusTool(tuner, self.llm), TuneServerTool(tuner, self.llm)]
        self.service = FriendService(cfg, self.registry, self.llm, self.memory, self.log, self.console)
        if self.registry.advisors():
            tools.append(AskFriendTool(self.service))
        if self.registry.workers() and cfg.max_depth >= 1:
            tools.append(DelegateTool(self.service))
        self.pool = ToolPool(tools, cfg.max_depth)
        self.service.tool_pool = self.pool          # workers pick their own tools from the pool

        self.messages: list = []

    def setup(self, *, new: bool = False, sync: bool = False) -> None:
        """Prepare the sandbox and load (or reset) the saved conversation."""
        if new:                                     # start a fresh conversation
            self.sessions.delete()
            self.project_memory.delete()
            self.memory.delete()
        if self.cfg.sandbox_mode == "bwrap":
            action = self.devcopy.sync(force=sync)
            if action:
                msg = f"[dev copy] {action}: {self.devcopy.dest}"
                if sync:                                    # --sync was asked for, so say so
                    self.console.info(msg)
                else:
                    self.console.debug(0, msg)
        elif sync:
            self.console.info("[note] --sync only applies in bwrap mode")
        self.sandbox.start()
        self.memory.load()
        self.messages = self.sessions.load(self.system)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self.system   # resumed sessions see the current setup (e.g. --friends)
        self._inject_project_memory()
        history = sum(1 for m in self.messages if m.get("role") != "system")    # the memory note is not history
        self.console.debug(0, f"[session] {history} earlier messages loaded" if history
                           else "[session] new conversation")
        self._recap = self._make_recap()
        self.llm.wait_ready()
        self.console.debug(0, f"[model] {self.llm.model_name}")
        if self.registry:
            self.console.debug(0, f"[friends] {', '.join(self.registry.names())}")

    def _make_recap(self) -> str:
        """What the resumed conversation was about, so the user is reminded where they left off."""
        if len(self.messages) < 2:
            return ""
        try:
            age = time.time() - os.path.getmtime(self.sessions.path)
        except OSError:
            age = None
        width = max(40, min(shutil.get_terminal_size().columns - 14, 140))
        return format_recap(self.messages, age, width=width)

    def _inject_project_memory(self) -> None:
        """Refresh a single system note; never accumulate duplicate memory notes."""
        prefix = "[persistent project memory]"
        self.messages[:] = [m for m in self.messages if not
                            (m.get("role") == "system" and str(m.get("content", "")).startswith(prefix))]
        saved = self.project_memory.read()
        if saved:
            self.messages.insert(1, {"role": "system", "content": prefix +
                                   "\nHistorical excerpts (verify against files before acting):\n" + saved})

    def run_goal(self, goal: str) -> RunResult:
        """Run one goal and report the outcome. History is kept, so earlier work stays in context."""
        if self._recap:                     # a one-shot goal on a resumed conversation: --debug shows the reminder
            self.console.debug(0, self._recap)
            self._recap = ""
        self._inject_project_memory()
        self.messages.append({"role": "user", "content": f"GOAL: {goal}"})
        self.console.start_clock()
        self.log.log("start", goal=goal, model=self.llm.model_name)
        agent = Agent("main", self.llm, self.pool.select(MAIN_TOOLS, 0), self.cfg.max_steps,
                      self.log, self.console, on_save=lambda: self.sessions.save(self.messages),
                      on_drop=self.project_memory.remember)
        result = agent.run(self.messages)
        self.console.debug(0, f"[elapsed] {self.console.elapsed()}")
        if not self.cfg.debug:          # with --debug the steps already showed the outcome
            self._report(result)
        return result

    def _report(self, result: RunResult) -> None:
        """Quiet mode: the final result on stdout, or a short explanation on stderr."""
        if result.status == FINISHED:
            self.console.info(result.text or "(finished without an answer)")
            return
        if result.status == MAX_STEPS:
            msg = f"Stopped: hit the step limit ({self.cfg.max_steps}) without finishing."
            if result.text:
                msg += f"\nLast message: {result.text}"
        else:           # LLM_FAILED or CONTEXT_FULL: the text says why and what to do
            msg = f"Stopped: {result.text or 'the model server kept failing'}.\n(details: {self.cfg.log_file})"
        self.console.error(msg + "\nRun again with --debug to see every step.")

    def chat(self) -> None:
        """Interactive loop: one goal per prompt, until exit/quit/Ctrl-D.

        Messages typed while a goal runs are queued (see inbox.py) and run, in order, as soon as
        the goal is done, without showing the prompt in between."""
        if self._recap:                     # resumed: remind the user where they left off
            self.console.info(self._recap)
            self._recap = ""
        inbox = Inbox(self.console)
        if inbox.supported:
            self.console.info(TIP)
        while True:
            goal = inbox.pop()
            if goal is not None:
                self.console.info(f"\ngoal (queued)> {goal}")
            else:
                try:
                    with inbox.prefilled():
                        goal = input("\ngoal> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
            if goal in ("exit", "quit"):
                return
            if goal == "/queue":
                self.console.info(inbox.listing())
            elif goal == "/drop":
                self.console.info(f"[dropped {inbox.clear()} queued message(s)]")
            elif goal:
                with inbox.capturing():
                    self.run_goal(goal)
