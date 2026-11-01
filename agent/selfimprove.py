"""Promoting the model's improvements of its own code, safely.

The model edits a copy of the code (<workspace>/agent_dev). Code that is promoted into the real
tree runs OUTSIDE the sandbox from the next start on, so promotion is a gate, not a copy:

  1. the file must not be PROTECTED (it defines the jail, this gate, the tools or the wiring)
  2. it must not add a new risky capability (see `risk_increase`): new subprocess/socket/open
     calls, os.system, eval and so on. That is a heuristic against accidents and blunt misuse,
     NOT a security boundary against a determined model; the boundary is 1, 3 and 4
  3. compile + the test suite must pass in the dev copy, inside the jail
  4. after copying, the test suite and `-h` must pass on the new real tree, again inside the
     jail (the real code is shown to the jail read-only), so new code is never first run outside it

Anything that fails a check is queued for a human (`--promote`). Before changing anything the
old files are backed up, and a failed promotion is rolled back and remembered, so the model
cannot retry the same broken change in a loop. `--rollback` undoes the last promotion.

This module uses only the standard library on purpose: it must keep working, and be able to
roll back, even if the model has broken the rest of the package.

    python -m agent.selfimprove status|rollback
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
DEV_NAME = "agent_dev"                          # must match config.DEV_NAME (a test checks)
ITEMS = ("agent_local.py", "agent", "tests")    # must match devcopy.ITEMS (a test checks)

# Never promoted automatically: whoever controls these controls the jail, the host-side tools
# or what runs first. A human reviews changes to them with --promote.
PROTECTED = frozenset({
    "agent_local.py", "agent/__init__.py", "agent/cli.py", "agent/app.py", "agent/config.py",
    "agent/sandbox.py", "agent/tools.py", "agent/devcopy.py",
    "agent/selfimprove.py", "agent/selftools.py", "agent/hostcontrol.py",
})

MAX_AUTO_FILES = 12                             # a bigger change is for a human to read

JAIL_DEV = f"/workspace/{DEV_NAME}"
JAIL_REPO = "/repo"
_PYENV = "PYTHONDONTWRITEBYTECODE=1"
VERIFY_DEV = f"cd {JAIL_DEV} && {_PYENV} python -m compileall -q . && {_PYENV} python -m unittest discover -s tests -q"
VERIFY_HOST = (f"cd {JAIL_REPO} && {_PYENV} python -m unittest discover -s tests -q "
               f"&& {_PYENV} python agent_local.py -h >/dev/null")

# (command, timeout seconds, read-only extra binds) -> CompletedProcess-like (returncode, stdout, stderr)
Runner = Callable[[str, int, tuple], object]


# ---- the static check ---------------------------------------------------------------------
_RISKY_MODULES = {"subprocess", "socket", "ctypes", "importlib", "pickle", "marshal", "shutil",
                  "urllib", "http", "requests", "multiprocessing", "pty", "signal", "resource"}
_RISKY_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "breakpoint"}
_RISKY_ATTRS = {
    "os": {"system", "popen", "execv", "execve", "execvp", "execl", "execlp", "spawnl", "spawnv",
           "fork", "kill", "killpg", "remove", "unlink", "rmdir", "removedirs", "rename", "chmod",
           "chown", "symlink", "link", "environ", "putenv"},
    "sys": {"modules", "settrace", "setprofile"},
}
_ESCAPE_DUNDERS = {"__class__", "__subclasses__", "__globals__", "__builtins__", "__code__",
                   "__bases__", "__mro__"}


def _risk_counts(tree: ast.AST) -> Counter:
    c: Counter = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in _RISKY_MODULES:
                    c[f"import {a.name.split('.')[0]}"] += 1
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _RISKY_MODULES:
                c[f"import {node.module.split('.')[0]}"] += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _RISKY_BUILTINS:
                c[f"call {node.func.id}()"] += 1
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.attr in _RISKY_ATTRS.get(node.value.id, ()):
                c[f"use {node.value.id}.{node.attr}"] += 1
            elif isinstance(node.value, ast.Name) and node.value.id == "subprocess":
                c["use subprocess"] += 1
            if node.attr in _ESCAPE_DUNDERS:
                c[f"use {node.attr}"] += 1
        elif isinstance(node, ast.Name) and node.id in ("__builtins__",):
            c["use __builtins__"] += 1
    return c


def risk_increase(old_src: str, new_src: str) -> list[str]:
    """What the new version does that the old one did not: more of any risky construct.
    Raises SyntaxError if `new_src` does not parse. Existing uses are fine; only additions count."""
    new = _risk_counts(ast.parse(new_src))
    old = _risk_counts(ast.parse(old_src)) if old_src.strip() else Counter()
    return [f"adds {k} (+{n - old[k]})" for k, n in sorted(new.items()) if n > old[k]]


# ---- data ---------------------------------------------------------------------------------
@dataclass
class Change:
    path: str                   # relative to the code root, e.g. agent/llm.py
    kind: str                   # modified | added | deleted
    old: str
    new: str
    verdict: str = "auto"       # auto | protected | risky | deleted | rejected | unparsable
    reasons: list[str] = field(default_factory=list)

    @property
    def digest(self) -> str:
        return hashlib.sha256(f"{self.path}\0{self.new}".encode()).hexdigest()[:16]

    def diff(self) -> str:
        return "".join(difflib.unified_diff(self.old.splitlines(True), self.new.splitlines(True),
                                            f"{self.path} (host)", f"{self.path} (model)"))


@dataclass
class Result:
    applied: list[str] = field(default_factory=list)
    queued: list[tuple[str, str]] = field(default_factory=list)      # (path, why a human must look)
    failed: str = ""            # why nothing was (or everything was undone), "" if all went well
    nothing: bool = False       # no differences at all

    def text(self) -> str:
        if self.nothing:
            return "No changes: the dev copy is identical to the real code."
        out = []
        if self.applied:
            out.append("PROMOTED (active from the next start): " + ", ".join(self.applied))
        if self.failed:
            out.append("NOT PROMOTED: " + self.failed)
        if self.queued:
            out.append("Needs a human (run `python agent_local.py --promote`):\n" +
                       "\n".join(f"  {p}: {why}" for p, why in self.queued))
        return "\n".join(out) or "Nothing to promote."


@dataclass
class _Ran:
    returncode: int
    stdout: str = ""
    stderr: str = ""


def _tail(proc, n: int = 1500) -> str:
    text = ((getattr(proc, "stdout", "") or "") + (getattr(proc, "stderr", "") or "")).strip()
    return text[-n:]


class Promoter:
    def __init__(self, root: Path, dev: Path, runner: Runner, state_dir: Path | None = None,
                 protected: frozenset[str] = PROTECTED):
        self.root, self.dev, self.runner = Path(root), Path(dev), runner
        self.state = Path(state_dir) if state_dir else self.root / ".selfimprove"
        self.protected = protected

    # -- what differs ----------------------------------------------------------------------
    @staticmethod
    def _py_files(base: Path) -> set[str]:
        found: set[str] = set()
        for item in ITEMS:
            p = base / item
            if p.is_file():
                found.add(item)
            elif p.is_dir():
                found |= {f.relative_to(base).as_posix() for f in p.rglob("*.py") if "__pycache__" not in f.parts}
        return found

    def changes(self) -> list[Change]:
        if not self.dev.exists():
            raise FileNotFoundError("No dev copy yet. Run the agent once (or use --sync).")
        rejected = self._rejected()
        out = []
        for rel in sorted(self._py_files(self.root) | self._py_files(self.dev)):
            a, b = self.root / rel, self.dev / rel
            old = a.read_text() if a.exists() else ""
            new = b.read_text() if b.exists() else ""
            if a.exists() and b.exists() and old == new:
                continue
            kind = "added" if not a.exists() else "deleted" if not b.exists() else "modified"
            out.append(self._judge(Change(rel, kind, old, new), rejected))
        return out

    def _judge(self, c: Change, rejected: set[str]) -> Change:
        if c.path in self.protected:
            c.verdict, c.reasons = "protected", ["protected: it defines the jail, the host-side tools or the wiring"]
        elif c.kind == "deleted":
            c.verdict, c.reasons = "deleted", ["deleting a file needs a human"]
        else:
            try:
                reasons = risk_increase(c.old, c.new)
            except SyntaxError as e:
                c.verdict, c.reasons = "unparsable", [f"does not parse: {e.msg} (line {e.lineno})"]
                return c
            if reasons:
                c.verdict, c.reasons = "risky", reasons
            elif c.digest in rejected:
                c.verdict, c.reasons = "rejected", ["this exact change already failed verification and was rolled back"]
        return c

    def _run(self, command: str, binds: tuple = ()):
        """Run a verification command in the jail. A timeout or a failure to run counts as a
        failed check, never as a pass."""
        try:
            return self.runner(command, 300, binds)
        except subprocess.TimeoutExpired:
            return _Ran(1, stderr="timed out after 300s")
        except OSError as e:
            return _Ran(1, stderr=f"could not run the check: {e}")

    # -- promoting -------------------------------------------------------------------------
    def promote(self, auto: bool = True, approve_all: bool = False) -> Result:
        """Promote what may be promoted. auto=False only reports (queue mode).
        approve_all=True is for the human running --promote: every parsable change goes."""
        changes = self.changes()
        if not changes:
            return Result(nothing=True)
        if approve_all:
            chosen = [c for c in changes if c.verdict != "unparsable"]
        elif auto:
            chosen = [c for c in changes if c.verdict == "auto"]
            if len(chosen) > MAX_AUTO_FILES:
                for c in chosen:
                    c.reasons = [f"more than {MAX_AUTO_FILES} files changed at once: too big to promote without review"]
                chosen = []
        else:
            chosen = []
        chosen_paths = {c.path for c in chosen}
        queued = [(c.path, "; ".join(c.reasons) or "waiting for approval (queue mode)")
                  for c in changes if c.path not in chosen_paths]
        if not auto and not approve_all:
            return self._finish(Result(queued=queued))
        if not chosen:
            return self._finish(Result(queued=queued))

        # 3. tests in the dev copy, in the jail
        proc = self._run(VERIFY_DEV)
        if proc.returncode != 0:
            return self._finish(Result(queued=queued, failed="compile/tests failed in the dev copy:\n" + _tail(proc)))

        # backup, apply, 4. test the new real tree in the jail; on ANY problem, put the old files back
        backup = self._backup(chosen)
        try:
            for c in chosen:
                target = self.root / c.path
                if c.kind == "deleted":
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(c.new)
            proc = self._run(VERIFY_HOST, ((str(self.root), JAIL_REPO),))
        except BaseException:
            self._restore(backup)
            raise
        if proc.returncode != 0:
            self._restore(backup)
            self._remember_rejected(chosen)
            return self._finish(Result(queued=queued, failed=(
                "the new code failed its tests when put in place, so it was rolled back "
                "(and will not be tried again):\n" + _tail(proc))))

        res = Result(applied=[c.path for c in chosen], queued=queued)
        self._audit("promoted", chosen, backup)
        return self._finish(res)

    def _finish(self, res: Result) -> Result:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "pending.json").write_text(json.dumps(
            [{"path": p, "why": why} for p, why in res.queued], indent=1))
        return res

    def selftest(self) -> tuple[bool, str]:
        """Run both verification steps on the CURRENT code, changing nothing: shows whether the
        jail-based checks work on this machine before auto-promotion is turned on."""
        ok, lines = True, []
        steps = [("real code (tests and -h, in the jail)", VERIFY_HOST, ((str(self.root), JAIL_REPO),))]
        if self.dev.exists():
            steps.append(("dev copy (compile and tests, in the jail)", VERIFY_DEV, ()))
        for label, command, binds in steps:
            proc = self._run(command, binds)
            lines.append(f"{label}: {'ok' if proc.returncode == 0 else 'FAILED'}")
            if proc.returncode != 0:
                ok = False
                lines.append(_tail(proc))
        return ok, "\n".join(lines)

    # -- backups, rollback, memory ---------------------------------------------------------
    def _backup(self, chosen: list[Change]) -> Path:
        d = self.state / "backups" / time.strftime("%Y%m%d-%H%M%S")
        n = 1
        while d.exists():
            n += 1
            d = d.with_name(f"{d.name.split('+')[0]}+{n}")
        d.mkdir(parents=True)
        files = []
        for c in chosen:
            src = self.root / c.path
            existed = src.exists()
            if existed:
                (d / "files" / c.path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, d / "files" / c.path)
            files.append({"path": c.path, "existed": existed})
        (d / "manifest.json").write_text(json.dumps({"files": files, "rolled_back": False}, indent=1))
        return d

    def _restore(self, backup: Path) -> list[str]:
        manifest = json.loads((backup / "manifest.json").read_text())
        restored = []
        for f in manifest["files"]:
            target = self.root / f["path"]
            if f["existed"]:
                shutil.copy2(backup / "files" / f["path"], target)
            else:
                target.unlink(missing_ok=True)
            restored.append(f["path"])
        manifest["rolled_back"] = True
        (backup / "manifest.json").write_text(json.dumps(manifest, indent=1))
        return restored

    def rollback(self) -> list[str]:
        """Undo the most recent promotion that has not been undone. Returns the files restored."""
        base = self.state / "backups"
        for d in sorted(base.iterdir(), reverse=True) if base.exists() else []:
            m = json.loads((d / "manifest.json").read_text())
            if not m["rolled_back"]:
                restored = self._restore(d)
                self._remember_rejected_paths(restored)
                self._audit("rolled back by request", [], d)
                return restored
        return []

    def _rejected(self) -> set[str]:
        f = self.state / "rejected.json"
        return set(json.loads(f.read_text())) if f.exists() else set()

    def _remember_rejected(self, chosen: list[Change]) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "rejected.json").write_text(json.dumps(sorted(self._rejected() | {c.digest for c in chosen})))

    def _remember_rejected_paths(self, paths: list[str]) -> None:
        """After a manual rollback, the dev copy still holds the change: mark it so the model
        does not re-promote it automatically."""
        try:
            changes = [c for c in self.changes() if c.path in paths]
        except FileNotFoundError:
            return
        self._remember_rejected(changes)

    def _audit(self, what: str, chosen: list[Change], backup: Path) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "backup": backup.name,
                 "files": [c.path for c in chosen],
                 "diff": "".join(c.diff() for c in chosen)[:20000]}
        with open(self.state / "log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")


def review_and_promote(promoter: Promoter, ask=input, out=print, interactive: bool = True) -> int:
    """The human's side (`--promote`): show everything the model changed and what automatic
    promotion would have done with it, ask, then promote it all after the same verification."""
    try:
        changes = promoter.changes()
    except FileNotFoundError as e:
        out(str(e))
        return 1
    if not changes:
        out("No differences.")
        return 0
    out(f"{len(changes)} changed file(s):")
    for c in changes:
        out(f"  {c.verdict:10} {c.kind:9} {c.path}" + (f"   ({'; '.join(c.reasons)})" if c.reasons else ""))
    out("")
    for c in changes:
        out(c.diff().rstrip("\n") or f"(no textual difference in {c.path})")
        out("")
    bad = [c.path for c in changes if c.verdict == "unparsable"]
    if bad:
        out("Not applied, they do not parse: " + ", ".join(bad))
    if not interactive:
        out("--promote needs a terminal: it asks you to confirm.")
        return 1
    n = len(changes) - len(bad)
    if n == 0 or ask(f"Verify in the jail, then apply {n} change(s)? [y/N] ").strip().lower() != "y":
        out("Nothing applied.")
        return 0
    result = promoter.promote(auto=False, approve_all=True)
    out(result.text())
    return 1 if result.failed else 0


def main(argv: list[str]) -> int:
    """Standalone commands that work even if the rest of the package is broken."""
    from_env = os.environ.get("WORKSPACE") or str(ROOT / "workspace")
    p = Promoter(ROOT, Path(from_env) / DEV_NAME, runner=lambda *a: None)
    cmd = argv[0] if argv else "status"
    if cmd == "rollback":
        restored = p.rollback()
        print("Restored: " + ", ".join(restored) if restored else "Nothing to roll back.")
        return 0
    if cmd == "status":
        log = p.state / "log.jsonl"
        if not log.exists():
            print("No promotions yet.")
            return 0
        for line in log.read_text().splitlines()[-10:]:
            e = json.loads(line)
            print(f"{e['time']}  {e['what']}: {', '.join(e['files']) or '-'}  (backup {e['backup']})")
        return 0
    print("usage: python -m agent.selfimprove [status|rollback]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
