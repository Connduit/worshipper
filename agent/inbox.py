"""Messages typed while the agent is working are queued and run afterwards.

At the prompt, input() is fine. While a goal runs, the terminal is put in no-echo mode and this
module reads the keystrokes itself: what you type shows at the end of the live status line,
Enter queues it (a note confirms it) and the chat loop runs the queue in order once the current
goal is done. `/queue` lists what is waiting, `/drop` clears it.

Without a terminal (a pipe, a test) nothing is captured and input() works as before.
"""
from __future__ import annotations

import codecs
import contextlib
import os
import select
import sys
import threading
from collections import deque

from .stepinfo import short

try:                                    # not on Windows
    import termios
    import tty
except ImportError:                     # pragma: no cover
    termios = tty = None

TIP = "(you can keep typing while it works: Enter queues a message, /queue lists them, /drop clears them)"


class Inbox:
    MAX = 20                            # queued messages
    MAX_TYPED = 500                     # characters in the line being typed

    def __init__(self, console, stdin=None):
        self.console = console
        self._stdin = stdin if stdin is not None else sys.stdin
        self._items: deque[str] = deque()
        self._buf = ""
        self._esc = ""                  # "" | "esc" | "csi": swallowing an escape sequence (arrow keys...)
        self._lock = threading.RLock()
        self.leftover = ""              # typed but not sent when a run ended; offered at the next prompt

    # -- the queue ------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def pop(self) -> str | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def clear(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n

    def listing(self) -> str:
        items = self.items()
        if not items:
            return "[queue is empty]"
        return "[queued]\n" + "\n".join(f"  {i}. {short(t, 70)}" for i, t in enumerate(items, 1))

    def submit(self, text: str) -> None:
        """A finished line, from the keyboard (or a test): a command, or a message to queue."""
        text = text.strip()
        if not text:
            return
        if text == "/queue":
            self.console.note(self.listing())
        elif text == "/drop":
            self.console.note(f"[dropped {self.clear()} queued message(s)]")
        else:
            with self._lock:
                if len(self._items) >= self.MAX:
                    self.console.note(f"[queue full ({self.MAX}): not queued] {short(text, 50)}")
                    return
                self._items.append(text)
                n = len(self._items)
            self.console.note(f"[queued {n}] {short(text, 60)}")
        self.console.refresh_status()

    # -- keystrokes -----------------------------------------------------------------------------
    def feed(self, chars: str) -> None:
        """Handle typed characters one by one (a tiny line editor: Enter, Backspace, Ctrl-U/W)."""
        for ch in chars:
            self._key(ch)
        self.console.refresh_status()

    def _key(self, ch: str) -> None:
        if self._esc == "esc":                      # after ESC: "[" or "O" starts an arrow-key style sequence
            self._esc = "csi" if ch in "[O" else ""
            return
        if self._esc == "csi":                      # ends with a final byte 0x40-0x7e
            if "@" <= ch <= "~":
                self._esc = ""
            return
        if ch == "\x1b":
            self._esc = "esc"
        elif ch in "\r\n":
            line, self._buf = self._buf, ""
            self.submit(line)
        elif ch in "\x7f\x08":
            self._buf = self._buf[:-1]
        elif ch == "\x15":                          # Ctrl-U
            self._buf = ""
        elif ch == "\x17":                          # Ctrl-W: the last word
            self._buf = self._buf.rstrip().rsplit(" ", 1)[0] + " " if " " in self._buf.rstrip() else ""
        elif ch >= " " and ch != "\x7f" and len(self._buf) < self.MAX_TYPED:
            self._buf += ch

    def _suffix(self) -> str:
        """What the status line shows after the timers: the queue size and the line being typed."""
        parts = []
        if self._items:
            parts.append(f"{len(self._items)} queued")
        parts.append("> " + self._buf + "_")
        return " | ".join(parts)

    # -- capturing ------------------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return bool(termios and self._stdin.isatty() and sys.stderr.isatty())

    @contextlib.contextmanager
    def capturing(self):
        """While the body runs, read keystrokes into the queue. A no-op without a terminal."""
        if not self.supported:
            yield self
            return
        fd = self._stdin.fileno()
        saved = termios.tcgetattr(fd)
        stop = threading.Event()
        thread = threading.Thread(target=self._read, args=(fd, stop), daemon=True)
        with self._lock:                            # a half-typed line survives a run that was queued behind another
            self._buf, self.leftover = self.leftover, ""
        try:
            tty.setcbreak(fd)                       # no echo, no line buffering; Ctrl-C still works
            self.console.set_status_suffix(self._suffix)
            thread.start()
            yield self
        finally:
            stop.set()
            thread.join(timeout=1)
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self.console.set_status_suffix(None)
            self.console.clear_status()
            with self._lock:
                self.leftover, self._buf, self._esc = self._buf, "", ""

    def _read(self, fd: int, stop: threading.Event) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        while not stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                self._esc = ""                      # a lone Esc key must not swallow the next character
                continue
            try:
                data = os.read(fd, 4096)
            except OSError:
                return
            if not data:
                return
            self.feed(decoder.decode(data))

    @contextlib.contextmanager
    def prefilled(self):
        """For the next input(): start with the text that was typed but not sent."""
        text, self.leftover = self.leftover, ""
        try:
            import readline
        except ImportError:
            readline = None
        if text and readline:
            readline.set_startup_hook(lambda: readline.insert_text(text))
        try:
            yield
        finally:
            if text and readline:
                readline.set_startup_hook()
