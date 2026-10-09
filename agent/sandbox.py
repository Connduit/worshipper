"""Where the model's shell commands run.

A Sandbox turns a command string into an argv and runs it. Three flavours:
  BwrapSandbox   bubblewrap jail (default): read-only system, no network
  DockerSandbox  `docker exec` into an existing container
  NoSandbox      run directly; used by the model *inside* the jail to test its edits
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Config


class Sandbox(ABC):
    def start(self) -> None:
        """Set up anything needed before the first command (default: nothing)."""

    @abstractmethod
    def command(self, command: str, readonly: bool = False) -> list[str]:
        """argv that runs `command` inside the sandbox.

        readonly=True asks for /workspace to be read-only (only BwrapSandbox enforces it).
        """

    def run(self, command: str, timeout: int, readonly: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(self.command(command, readonly),
                              capture_output=True, text=True, timeout=timeout)


class NoSandbox(Sandbox):
    """We're already inside the jail (dev/test role), so run commands directly."""

    def command(self, command: str, readonly: bool = False) -> list[str]:
        return ["bash", "-c", command]


class DockerSandbox(Sandbox):
    def __init__(self, container: str):
        self.container = container

    def command(self, command: str, readonly: bool = False) -> list[str]:
        return ["docker", "exec", self.container, "bash", "-lc", command]


class LlamaBridge:
    """Host side: forward a private unix socket to the llama.cpp TCP port (via socat).

    Inside the jail a second socat re-listens on loopback:PORT, so the model can
    reach llama-server and nothing else.
    """

    def __init__(self, port: int):
        self.port = port
        self.sock_dir: str | None = None

    def start(self) -> None:
        if shutil.which("socat") is None:
            sys.exit("socat not found on the host. Install it: sudo pacman -S socat")
        self.sock_dir = tempfile.mkdtemp(prefix="agent-sock-")      # mkdtemp is mode 0700
        sock = os.path.join(self.sock_dir, "llama.sock")
        p = subprocess.Popen(
            ["socat", f"UNIX-LISTEN:{sock},fork,mode=600", f"TCP:127.0.0.1:{self.port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        atexit.register(p.terminate)
        atexit.register(shutil.rmtree, self.sock_dir, True)
        for _ in range(20):                                         # wait up to ~2s for the socket
            if os.path.exists(sock):
                return
            time.sleep(0.1)
        print("[warn] llama bridge socket did not appear; the sandbox won't reach llama-server")


class BwrapSandbox(Sandbox):
    def __init__(self, workspace: Path | str, bridge: LlamaBridge | None = None):
        self.workspace = str(workspace)
        self.bridge = bridge

    def start(self) -> None:
        if self.bridge:
            self.bridge.start()

    def command(self, command: str, readonly: bool = False) -> list[str]:
        extra: list[str] = []
        if self.bridge and self.bridge.sock_dir:
            # Only the socket dir is exposed. Inside, socat re-listens on loopback:PORT.
            extra = ["--ro-bind", self.bridge.sock_dir, "/run/llama"]
            command = (
                f"socat TCP-LISTEN:{self.bridge.port},bind=127.0.0.1,fork,reuseaddr "
                f"UNIX-CONNECT:/run/llama/llama.sock >/dev/null 2>&1 &\n"
                f"sleep 0.3\n{command}"
            )

        # Read-only system, writable /workspace only. --unshare-all cuts off network,
        # other processes, IPC, etc. (Deliberately NO --share-net.)
        return [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/bin", "/sbin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--ro-bind" if readonly else "--bind", self.workspace, "/workspace",
            *extra,
            "--chdir", "/workspace",
            "--setenv", "HOME", "/workspace",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "bash", "-c", command,
        ]


def make_sandbox(cfg: Config) -> Sandbox:
    if cfg.sandbox_mode == "none":
        return NoSandbox()
    if cfg.sandbox_mode == "docker":
        return DockerSandbox(cfg.container)
    bridge = LlamaBridge(cfg.llama_port) if cfg.expose_llama else None
    return BwrapSandbox(cfg.workspace, bridge)      # anything else gets the safe default
