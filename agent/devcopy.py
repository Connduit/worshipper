"""The model's editable copy of this code.

The model never edits the code that builds the jail. On first run (bwrap mode) the
entry script, the package and the tests are copied into <workspace>/agent_dev/; the
model edits those. You review with --diff and promote by hand.
"""
from __future__ import annotations

import difflib
import shutil
from pathlib import Path

ITEMS = ("agent_local.py", "agent", "tests")     # what gets copied, relative to the code root
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


class DevCopy:
    def __init__(self, source_root: Path, dest: Path):
        self.source_root = source_root
        self.dest = dest

    def sync(self, force: bool = False) -> None:
        """Create the dev copy, or (force) wipe it and start over from the host code."""
        existed = self.dest.exists()
        if existed and not force:
            return
        if existed:
            shutil.rmtree(self.dest)
        self.dest.mkdir(parents=True)
        for item in ITEMS:
            src = self.source_root / item
            if src.is_dir():
                shutil.copytree(src, self.dest / item, ignore=IGNORE)
            elif src.is_file():
                shutil.copy2(src, self.dest / item)
        print(f"[dev copy] {'reset' if existed else 'created'}: {self.dest}")

    def diff(self) -> list[str]:
        """Unified diff of every .py file, host vs. the model's copy."""
        if not self.dest.exists():
            raise FileNotFoundError("No dev copy yet. Run the agent once (or use --sync).")
        rel = sorted(self._py_files(self.source_root) | self._py_files(self.dest))
        out: list[str] = []
        for r in rel:
            a, b = self.source_root / r, self.dest / r
            a_lines = a.read_text().splitlines(True) if a.exists() else []
            b_lines = b.read_text().splitlines(True) if b.exists() else []
            out += difflib.unified_diff(a_lines, b_lines, f"{r} (host)", f"{r} (model)")
        return out

    @staticmethod
    def _py_files(root: Path) -> set[str]:
        found = set()
        for item in ITEMS:
            p = root / item
            if p.is_file():
                found.add(item)
            elif p.is_dir():
                found |= {str(f.relative_to(root)) for f in p.rglob("*.py")
                          if "__pycache__" not in f.parts}
        return found
