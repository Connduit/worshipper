"""App: builds all the pieces and runs goals. The only place that knows about everything."""
from __future__ import annotations

from .agent import FINISHED, MAX_STEPS, Agent, RunResult
from .config import ROOT, Config
from .console import Console
from .devcopy import DevCopy
from .friends import AskFriendTool, DelegateTool, FriendRegistry, FriendService
from .llm import LLM
from .prompts import build_system
from .sandbox import make_sandbox
from .storage import EventLog, FriendMemory, SessionStore
from .tools import FinishTool, GetTimeTool, RunShellTool, ToolPool

MAIN_TOOLS = ("get_time", "run_shell", "ask_friend", "delegate")


class App:
    def __init__(self, cfg: Config, llm: LLM | None = None):
        cfg.workspace.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.console = Console(cfg.debug)
        self.log = EventLog(cfg.log_file)
        self.sessions = SessionStore(cfg.session_file)
        self.memory = FriendMemory(cfg.friend_memory_file, cfg.max_friend_history)
        self.devcopy = DevCopy(ROOT, cfg.dev_dir)
        self.sandbox = make_sandbox(cfg)
        self.llm = llm or LLM(cfg.llm_url, cfg.model, self.log, self.console)
        self.registry = FriendRegistry.load(cfg.friends_file, enabled=cfg.use_friends)
        self.system = build_system(cfg, self.registry)

        tools = [GetTimeTool(), RunShellTool(self.sandbox, cfg.default_timeout, cfg.max_timeout),
                 FinishTool()]
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
        self.console.debug(0, f"[session] {len(self.messages) - 1} earlier messages loaded"
                           if len(self.messages) > 1 else "[session] new conversation")
        if self.registry:
            self.console.debug(0, f"[friends] {', '.join(self.registry.names())}")

    def run_goal(self, goal: str) -> RunResult:
        """Run one goal and report the outcome. History is kept, so earlier work stays in context."""
        self.messages.append({"role": "user", "content": f"GOAL: {goal}"})
        self.log.log("start", goal=goal, model=self.cfg.model)
        agent = Agent("main", self.llm, self.pool.select(MAIN_TOOLS, 0), self.cfg.max_steps,
                      self.log, self.console, on_save=lambda: self.sessions.save(self.messages))
        result = agent.run(self.messages)
        if not self.cfg.debug:          # with --debug the steps already showed the outcome
            self._report(result)
        return result

    def _report(self, result: RunResult) -> None:
        """Quiet mode: the final result on stdout, or a short explanation on stderr."""
        if result.status == FINISHED:
            self.console.info(result.text or "(finished, no summary given)")
            return
        if result.status == MAX_STEPS:
            msg = f"Stopped: hit the step limit ({self.cfg.max_steps}) without finishing."
            if result.text:
                msg += f"\nLast message: {result.text}"
        else:
            msg = f"Stopped: the model server kept failing; see {self.cfg.log_file}."
        self.console.error(msg + "\nRun again with --debug to see every step.")

    def chat(self) -> None:
        """Interactive loop: one goal per prompt, until exit/quit/Ctrl-D."""
        while True:
            try:
                goal = input("\ngoal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if goal in ("exit", "quit"):
                return
            if goal:
                self.run_goal(goal)
