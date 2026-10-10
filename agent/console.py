"""All terminal output goes through here.

Quiet by default: stdout gets the final result, stderr gets problems. With --debug
every step (reasoning, tool calls, observations, friend chatter) is shown as well.
Steps are always recorded in the event log regardless.
"""
from __future__ import annotations

import sys


class Console:
    def __init__(self, debug: bool = False):
        self.debug_enabled = debug

    def info(self, text) -> None:
        """Always shown, on stdout: the result, or output of something the user asked for."""
        print(text)

    def error(self, text) -> None:
        """Always shown, on stderr: the run did not end well."""
        print(text, file=sys.stderr)

    def debug(self, depth: int, text) -> None:
        """Shown only with --debug, indented by nesting depth so a worker's steps stand out."""
        if not self.debug_enabled:
            return
        ind = "    " * depth
        print("\n".join(ind + line if line else line for line in str(text).split("\n")))
