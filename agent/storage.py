"""Everything that is written to disk: the event log, the session, friends' memory."""
from __future__ import annotations

import json
import os
import sys


class EventLog:
    """Append-only JSON-lines log of everything the agents do."""

    def __init__(self, path: str):
        self.path = path

    def log(self, event: str, **fields) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"event": event, **fields}) + "\n")


class SessionStore:
    """The main agent's conversation, so a run can be resumed."""

    def __init__(self, path: str):
        self.path = path

    def load(self, system_prompt: str) -> list:
        """Resume the previous conversation, or start a fresh one."""
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return [{"role": "system", "content": system_prompt}]

    def save(self, messages: list) -> None:
        with open(self.path, "w") as f:
            json.dump(messages, f)

    def delete(self) -> None:
        if os.path.exists(self.path):
            os.remove(self.path)


class FriendMemory:
    """Each advisor's persistent conversation with the agent, trimmed to `max_history`."""

    def __init__(self, path: str, max_history: int):
        self.path = path
        self.max_history = max_history
        self._data: dict[str, list] = {}

    def load(self) -> None:
        self._data = {}
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[warn] could not read {self.path}: {e}", file=sys.stderr)
            return
        if isinstance(data, dict):
            self._data = {k: v for k, v in data.items()
                          if isinstance(v, list) and all(isinstance(m, dict) for m in v)}

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f)

    def delete(self) -> None:
        self._data = {}
        if os.path.exists(self.path):
            os.remove(self.path)

    def history(self, name: str) -> list:
        return self._data.get(name, [])

    def prepare(self, name: str, system_prompt: str, message: str) -> list:
        """Return the advisor's history with its *current* system prompt and `message` added."""
        hist = self._data.setdefault(name, [])
        system_msg = {"role": "system", "content": system_prompt}
        if hist and hist[0].get("role") == "system":
            hist[0] = system_msg
        else:
            hist.insert(0, system_msg)
        hist.append({"role": "user", "content": message})
        return hist

    def commit(self, name: str, reply: str) -> None:
        """Record the advisor's reply, trim old messages, and save."""
        hist = self._data[name]
        hist.append({"role": "assistant", "content": reply})
        if len(hist) > self.max_history + 1:         # system message + the last N messages
            hist[1:] = hist[-self.max_history:]
        self.save()

    def rollback(self, name: str) -> None:
        """Drop the unanswered user message so user/assistant pairs stay intact."""
        self._data[name].pop()
