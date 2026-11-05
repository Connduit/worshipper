"""All terminal output goes through here.

Quiet by default: stdout gets the final result, stderr gets problems. With --debug
every step (reasoning, tool calls, observations, friend chatter) is shown as well.
Steps are always recorded in the event log regardless.

On a terminal, a one-line live status ("main, step 3: waiting for the model... 12s")
is redrawn in place on stderr and wiped before anything else is printed. It is never
written when stderr is a pipe or a file, so logs stay clean.
"""
from __future__ import annotations

import contextlib
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
        self._last_status = ""              # the text last drawn, for redrawing after other output
        self._suffix = None                 # () -> str shown at the end of the status line (queue input)
        self._clock: float | None = None

    # -- clock and live status ------------------------------------------------------------
    def start_clock(self) -> None:
        """Start the 'total' timer shown in the status line (one goal)."""
        self._clock = time.monotonic()

    def elapsed(self) -> str:
        return fmt_duration(time.monotonic() - self._clock) if self._clock is not None else ""

    def set_status_suffix(self, fn) -> None:
        """Text (from `fn()`) kept at the end of the status line, e.g. what the user is typing.
        None removes it. While one is set, the status line is redrawn after any other output."""
        with self._lock:
            self._suffix = fn

    def refresh_status(self) -> None:
        """Redraw the status line now (the suffix changed), instead of at the next tick."""
        with self._lock:
            if self._status_shown:
                self._draw(self._last_status)

    def status(self, text) -> bool:
        """Redraw the live status line in place. Returns False (and writes nothing) unless
        stderr is a terminal."""
        if not sys.stderr.isatty():
            return False
        if self._clock is not None:
            text = f"{text} | total {self.elapsed()}"
        with self._lock:
            self._draw(str(text))
        return True

    def _draw(self, text: str) -> None:    # caller holds the lock
        self._last_status = text
        avail = max(shutil.get_terminal_size().columns - 1, 10)     # never wrap, or it cannot be redrawn
        suffix = self._suffix() if self._suffix else ""
        if suffix:
            suffix = " | " + suffix
            if len(suffix) > avail // 2:
                suffix = " | ..." + suffix[-(avail // 2 - 6):]
            if len(text) + len(suffix) > avail:                      # the typed text wins over the status
                text = text[: max(avail - len(suffix) - 3, 0)] + "..."
        sys.stderr.write("\r\033[K" + (text + suffix)[:avail])
        sys.stderr.flush()
        self._status_shown = True

    def clear_status(self) -> None:
        with self._lock:
            self._wipe()

    def _wipe(self) -> bool:               # caller holds the lock; True if a status line was removed
        if self._status_shown:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self._status_shown = False
            return True
        return False

    def _maybe_redraw(self, was_shown: bool) -> None:      # caller holds the lock
        """While the user is typing into the status line it must not vanish after other output."""
        if was_shown and self._suffix is not None:
            self._draw(self._last_status)

    # -- normal output ----------------------------------------------------------------------
    def info(self, text) -> None:
        """Always shown, on stdout: the result, or output of something the user asked for."""
        with self._lock:
            again = self._wipe()
            print(text)
            self._maybe_redraw(again)

    def error(self, text) -> None:
        """Always shown, on stderr: the run did not end well."""
        with self._lock:
            again = self._wipe()
            print(text, file=sys.stderr)
            self._maybe_redraw(again)

    def note(self, text) -> None:
        """Always shown, on stderr: progress worth knowing about (e.g. waiting for the model)."""
        with self._lock:
            again = self._wipe()
            print(text, file=sys.stderr)
            self._maybe_redraw(again)

    def debug(self, depth: int, text) -> None:
        """Shown only with --debug, indented by nesting depth so a worker's steps stand out."""
        if not self.debug_enabled:
            return
        ind = "    " * depth
        with self._lock:
            again = self._wipe()
            print("\n".join(ind + line if line else line for line in str(text).split("\n")))
            self._maybe_redraw(again)


@contextlib.contextmanager
def working(console, text: str, depth: int = 0, *, tick: float = 1.0, heartbeat: float = 15.0):
    """While something slow runs (a model request, a long command) show `text` with a live timer:
    an in-place status line on a terminal, or (with --debug and no terminal) a log line every
    `heartbeat` seconds. The line is wiped when the block ends."""
    done = threading.Event()

    def beat():
        start = last_line = time.monotonic()
        while not done.wait(tick):
            now = time.monotonic()
            shown = console.status(f"{text} {fmt_duration(now - start)}")
            if not shown and now - last_line >= heartbeat:
                console.debug(depth, f"[{text} {fmt_duration(now - start)}]")
                last_line = now

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=2)
        console.clear_status()
