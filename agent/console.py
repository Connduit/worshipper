"""All terminal output goes through here.

Quiet by default: stdout gets the final result, stderr gets problems. With --debug
every step (reasoning, tool calls, observations, friend chatter) is shown as well.
Steps are always recorded in the event log regardless.

On a terminal, a one-line live status ("main, step 3: waiting for the model... 12s")
is redrawn in place on stderr and wiped before anything else is printed. It is never
written when stderr is a pipe or a file, so logs stay clean.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time

from .util import fmt_duration


class Console:
    def __init__(self, debug: bool = False):
        self.debug_enabled = debug
        self._lock = threading.RLock()      # the status line is redrawn from a timer thread
        self._status_shown = False
        self._clock: float | None = None

    # -- clock and live status ------------------------------------------------------------
    def start_clock(self) -> None:
        """Start the 'total' timer shown in the status line (one goal)."""
        self._clock = time.monotonic()

    def elapsed(self) -> str:
        return fmt_duration(time.monotonic() - self._clock) if self._clock is not None else ""

    def status(self, text) -> bool:
        """Redraw the live status line in place. Returns False (and writes nothing) unless
        stderr is a terminal."""
        if not sys.stderr.isatty():
            return False
        if self._clock is not None:
            text = f"{text} | total {self.elapsed()}"
        width = shutil.get_terminal_size().columns
        with self._lock:
            sys.stderr.write("\r\033[K" + str(text)[: max(width - 1, 10)])     # never wrap
            sys.stderr.flush()
            self._status_shown = True
        return True

    def clear_status(self) -> None:
        with self._lock:
            self._wipe()

    def _wipe(self) -> None:               # caller holds the lock
        if self._status_shown:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self._status_shown = False

    # -- normal output ----------------------------------------------------------------------
    def info(self, text) -> None:
        """Always shown, on stdout: the result, or output of something the user asked for."""
        with self._lock:
            self._wipe()
            print(text)

    def error(self, text) -> None:
        """Always shown, on stderr: the run did not end well."""
        with self._lock:
            self._wipe()
            print(text, file=sys.stderr)

    def note(self, text) -> None:
        """Always shown, on stderr: progress worth knowing about (e.g. waiting for the model)."""
        with self._lock:
            self._wipe()
            print(text, file=sys.stderr)

    def debug(self, depth: int, text) -> None:
        """Shown only with --debug, indented by nesting depth so a worker's steps stand out."""
        if not self.debug_enabled:
            return
        ind = "    " * depth
        with self._lock:
            self._wipe()
            print("\n".join(ind + line if line else line for line in str(text).split("\n")))
