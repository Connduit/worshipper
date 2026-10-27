"""Small shared helper."""
from __future__ import annotations


def truncate(s: str, n: int = 3000) -> str:
    return s if len(s) <= n else s[:n] + f"\n...[truncated {len(s) - n} chars]"


def fmt_duration(seconds: float) -> str:
    """45 -> '45s', 72 -> '1m 12s', 3725 -> '1h 02m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"
