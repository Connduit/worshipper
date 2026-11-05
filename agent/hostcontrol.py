"""Letting the model tune the llama-server it runs on, within hard bounds, with automatic undo.

The model cannot touch systemd or the host (its commands run in the jail). This module is the one
narrow door: it can change exactly three settings of the CURRENT model, by calling ./model.sh:

    ctx       context window per slot          4096 .. 65536
    np        parallel slots                      1 .. 4
    kv_cache  KV cache precision      f16 | q8_0 | q4_0

Nothing else: no other flags, no other model, no shell. Values are validated numbers and a fixed
choice list, so nothing the model writes ends up in a command line as free text. model.sh runs in
STRICT mode: after the restart it generates a few tokens and checks GPU memory, and it fails if
the server is slow or the GPU is nearly full. On any failure the previous settings are restored
automatically. At most MAX_CHANGES_PER_RUN changes happen per agent run.

Standard library only, like selfimprove.py.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

CTX_MIN, CTX_MAX = 4096, 65536          # per slot
NP_MIN, NP_MAX = 1, 4
KV_CHOICES = ("f16", "q8_0", "q4_0")
MAX_CHANGES_PER_RUN = 3
SWITCH_TIMEOUT = 1000                   # seconds model.sh may take (a load, maybe a download)

# model.sh sets these itself; anything else on the ExecStart line is kept as an "extra" flag
_MODEL_SH_FLAGS_WITH_VALUE = {"-ngl", "--host", "--port"}
_MODEL_SH_FLAGS = {"--jinja"}


@dataclass(frozen=True)
class Settings:
    model: str                  # what ./model.sh takes: org/repo[:quant] or a .gguf path
    ctx: int                    # per slot
    np: int
    kv_cache: str               # f16 | q8_0 | q4_0
    extra: tuple[str, ...] = () # other flags, kept exactly as they are

    @property
    def total(self) -> int:
        return self.ctx * self.np

    def summary(self) -> str:
        return (f"{self.ctx} tokens per slot x {self.np} slot(s) = {self.total} total, "
                f"KV cache {self.kv_cache}")


def parse_unit(text: str) -> Settings | None:
    """Read the current settings from the ExecStart line of deploy/llama.service."""
    line = next((l for l in text.splitlines() if l.startswith("ExecStart=")), None)
    if line is None:
        return None
    try:
        argv = shlex.split(line[len("ExecStart="):])
    except ValueError:
        return None
    model = total = None
    np_, ctk = 1, "f16"
    extra: list[str] = []
    i = 1                                           # argv[0] is the llama-server binary
    while i < len(argv):
        a = argv[i]
        nxt = argv[i + 1] if i + 1 < len(argv) else None
        if a in ("-hf", "-m") and nxt:
            model, i = nxt, i + 2
        elif a == "-c" and nxt and nxt.isdigit():
            total, i = int(nxt), i + 2
        elif a == "-np" and nxt and nxt.isdigit():
            np_, i = int(nxt), i + 2
        elif a in ("-ctk", "-ctv") and nxt:
            ctk, i = nxt, i + 2
        elif a in _MODEL_SH_FLAGS_WITH_VALUE and nxt:
            i += 2
        elif a in _MODEL_SH_FLAGS:
            i += 1
        else:                                       # an unknown flag, with its value(s) if any
            extra.append(a)
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                extra.append(argv[i])
                i += 1
    if not model or not total or np_ < 1:
        return None
    return Settings(model, max(total // np_, 1), np_, ctk if ctk in KV_CHOICES else "f16", tuple(extra))


def gpu_memory() -> tuple[int, int] | None:
    """(used, total) MiB of the first GPU, or None. nvidia-smi is not on PATH under WSL."""
    exe = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi"
                                         if os.access("/usr/lib/wsl/lib/nvidia-smi", os.X_OK) else None)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (int(x) for x in out.splitlines()[0].replace(" ", "").split(","))
        return used, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


Runner = Callable[[list, dict, int], object]    # (argv, env, timeout) -> CompletedProcess-like


def _run(argv: list, env: dict, timeout: int):
    return subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)


def _tail(proc, n: int = 1200) -> str:
    return (((getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")).strip())[-n:]


class Tuner:
    def __init__(self, root: Path, runner: Runner = _run, gpu: Callable = gpu_memory,
                 max_changes: int = MAX_CHANGES_PER_RUN):
        self.root = Path(root)
        self.unit = self.root / "deploy" / "llama.service"
        self.script = self.root / "model.sh"
        self.runner, self.gpu, self.max_changes = runner, gpu, max_changes
        self.changes_made = 0

    def current(self) -> Settings | None:
        try:
            return parse_unit(self.unit.read_text())
        except OSError:
            return None

    def status_text(self, window: int | None = None) -> str:
        s = self.current()
        lines = [f"model: {s.model}", f"settings: {s.summary()}"] if s else ["settings: unknown (no deploy/llama.service)"]
        if window:
            lines.append(f"context window per request, as the server reports it: {window} tokens")
        mem = self.gpu()
        if mem:
            used, total = mem
            lines.append(f"GPU memory: {used / 1024:.1f} / {total / 1024:.1f} GB ({used * 100 // total}% used)")
        lines.append(f"you may change: ctx {CTX_MIN}-{CTX_MAX} per slot, np {NP_MIN}-{NP_MAX}, kv_cache {'/'.join(KV_CHOICES)}"
                     f" (changes left this run: {max(self.max_changes - self.changes_made, 0)})")
        return "\n".join(lines)

    # -- changing settings ----------------------------------------------------------------------
    @staticmethod
    def _validate(ctx, np_, kv) -> str | None:
        """An error message, or None if every given value is acceptable."""
        for name, val, lo, hi in (("ctx", ctx, CTX_MIN, CTX_MAX), ("np", np_, NP_MIN, NP_MAX)):
            if val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, int):
                return f"{name} must be a whole number."
            if not lo <= val <= hi:
                return f"{name} must be between {lo} and {hi} (got {val})."
        if kv is not None and kv not in KV_CHOICES:
            return f"kv_cache must be one of {', '.join(KV_CHOICES)} (got {kv!r})."
        return None

    def apply(self, ctx: int | None = None, np: int | None = None, kv_cache: str | None = None) -> tuple[bool, str]:
        """Change the server's settings. Returns (success, message for the model)."""
        problem = self._validate(ctx, np, kv_cache)
        if problem:
            return False, problem
        if ctx is None and np is None and kv_cache is None:
            return False, "give at least one of ctx, np, kv_cache."
        if not self.script.exists():
            return False, "model.sh was not found, so the server cannot be restarted from here."
        cur = self.current()
        if cur is None:
            return False, "can't read the current server settings (deploy/llama.service), so nothing was changed."
        new = replace(cur, ctx=ctx or cur.ctx, np=np or cur.np, kv_cache=kv_cache or cur.kv_cache)
        if new == cur:
            return True, f"already set: {cur.summary()}. Nothing changed."
        if self.changes_made >= self.max_changes:
            return False, f"limit reached: at most {self.max_changes} server changes per run."
        self.changes_made += 1

        ok, out = self._switch(new, strict=True)
        self._audit(cur, new, ok)
        if ok:
            return True, f"server restarted with {new.summary()}.\n{out}"
        back_ok, _ = self._switch(cur, strict=False)
        return False, (f"{new.summary()} did not work:\n{out}\n" +
                       (f"The previous settings ({cur.summary()}) were restored automatically."
                        if back_ok else "WARNING: restoring the previous settings failed too; "
                                        "a human should run ./model.sh by hand."))

    def _switch(self, s: Settings, strict: bool) -> tuple[bool, str]:
        flags = ([] if s.kv_cache == "f16" else ["-ctk", s.kv_cache, "-ctv", s.kv_cache]) + list(s.extra)
        env = dict(os.environ, CTX=str(s.ctx), NP=str(s.np), EXTRA=" ".join(flags), WAIT="900")
        for k in ("STRICT", "NO_CHECK", "DRY_RUN", "SKIP_CHECK"):
            env.pop(k, None)
        env["STRICT" if strict else "NO_CHECK"] = "1"
        try:
            proc = self.runner(["bash", str(self.script), s.model], env, SWITCH_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, "model.sh timed out."
        except OSError as e:
            return False, f"could not run model.sh: {e}"
        return proc.returncode == 0, _tail(proc)

    def _audit(self, old: Settings, new: Settings, ok: bool) -> None:
        d = self.root / ".selfimprove"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "tune.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "from": old.summary(),
                                "to": new.summary(), "ok": ok}) + "\n")
