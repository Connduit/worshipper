"""Small shared helpers."""
from __future__ import annotations


def truncate(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def say(depth: int, text) -> None:
    """print(), indented by nesting depth so a worker's output is easy to tell apart."""
    ind = "    " * depth
    print("\n".join(ind + line if line else line for line in str(text).split("\n")))
