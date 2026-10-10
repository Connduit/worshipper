"""Small shared helper."""
from __future__ import annotations


def truncate(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"
