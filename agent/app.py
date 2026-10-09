"""App: builds all the pieces and runs goals. The only place that knows about everything."""
from __future__ import annotations

from .agent import Agent
from .config import ROOT, Config
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
        self.log = EventLog(cfg.log_file)
        self.sessions = SessionStore(cfg.session_file)
        self.memory = FriendMemory(cfg.friend_memory_file, cfg.max_friend_history)
        self.devcopy = DevCopy(ROOT, cfg.dev_dir)
        self.sandbox = make_sandbox(cfg)
        self.llm = llm or LLM(cfg.llm_url, cfg.model, self.log)
        self.registry = FriendRegistry.load(cfg.friends_file, enabled=cfg.use_friends)
        self.system = build_system(cfg, self.registry)

        tools = [GetTimeTool(), RunShellTool(self.sandbox, cfg.default_timeout, cfg.max_timeout),
                 FinishTool()]
        self.service = FriendService(cfg, self.registry, self.llm, self.memory, self.log)
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
            self.devcopy.sync(force=sync)
        elif sync:
            print("[note] --sync only applies in bwrap mode")
        self.sandbox.start()
        self.memory.load()
        self.messages = self.sessions.load(self.system)
        print(f"[session] {len(self.messages) - 1} earlier messages loaded"
              if len(self.messages) > 1 else "[session] new conversation")
        if self.registry:
            print(f"[friends] {', '.join(self.registry.names())}")

    def run_goal(self, goal: str):
        """Run one goal. History is kept, so earlier work stays in context."""
        self.messages.append({"role": "user", "content": f"GOAL: {goal}"})
        self.log.log("start", goal=goal, model=self.cfg.model)
        agent = Agent("main", self.llm, self.pool.select(MAIN_TOOLS, 0), self.cfg.max_steps,
                      self.log, on_save=lambda: self.sessions.save(self.messages))
        return agent.run(self.messages)

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
