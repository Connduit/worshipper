"""All settings in one place, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent   # the directory holding agent_local.py
DEV_NAME = "agent_dev"                          # the model's copy lives in <workspace>/agent_dev/


@dataclass(frozen=True)
class Config:
    llama_port: int
    llm_url: str
    model: str                  # llama-server mostly ignores this
    sandbox_mode: str           # "bwrap" | "docker" | "none"
    container: str              # docker mode only
    workspace: Path
    expose_llama: bool
    max_steps: int
    log_file: str
    session_file: str           # kept outside the sandbox
    use_friends: bool           # off by default: the agent works solo (--friends turns it on)
    friends_file: str
    max_depth: int              # how deep delegation may nest; 0 = no delegation
    default_timeout: int = 30
    max_timeout: int = 300
    max_friend_history: int = 12    # messages kept per advisor (even, so pairs stay intact)

    @property
    def friend_memory_file(self) -> str:
        return os.path.splitext(self.session_file)[0] + ".friends.json"

    @property
    def dev_dir(self) -> Path:
        return self.workspace / DEV_NAME

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        port = int(env.get("LLAMA_PORT", "8080"))
        workspace = env.get("WORKSPACE", str(ROOT / "workspace"))
        return cls(
            llama_port=port,
            llm_url=env.get("LLM_URL", f"http://127.0.0.1:{port}/v1"),
            model=env.get("LLM_MODEL", "local"),
            sandbox_mode=env.get("SANDBOX_MODE", "bwrap"),
            container=env.get("SANDBOX", "agent-sandbox"),
            workspace=Path(os.path.abspath(os.path.expanduser(workspace))),
            expose_llama=env.get("EXPOSE_LLAMA", "1") == "1",
            max_steps=int(env.get("MAX_STEPS", "25")),
            log_file=env.get("LOG_FILE", "agent_run.jsonl"),
            session_file=env.get("SESSION_FILE", "session.json"),
            use_friends=env.get("USE_FRIENDS", "0") == "1",
            friends_file=env.get("FRIENDS_FILE", "friends.json"),
            max_depth=int(env.get("MAX_DEPTH", "1")),
        )
