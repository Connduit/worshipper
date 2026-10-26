"""Talking to the llama.cpp server (OpenAI-compatible)."""
from __future__ import annotations

import contextlib
import threading
import time
from urllib import error as _urlerror, request as _urlreq

from .console import Console
from .storage import EventLog


AUTO_MODEL = ("", "auto", "local")   # "ask the server which model it is serving"


class LLM:
    """One chat-completions endpoint + model, with retries."""

    TEMPERATURES = (0.2, 0.7, 1.0)    # one per attempt: vary the sample on retries
    HEARTBEAT = 15.0                  # seconds between "still waiting" lines (--debug)

    def __init__(self, base_url: str, model: str, log: EventLog, console: Console,
                 api_key: str = "not-needed-for-local"):
        # Imported here so -h, --diff and --friends work without `openai` installed.
        from openai import APIConnectionError, APIStatusError, OpenAI
        self.base_url = base_url
        self.model = model
        self._log = log
        self._console = console
        self._errors = (APIStatusError, APIConnectionError)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._resolved: str | None = None

    @property
    def model_name(self) -> str:
        """The model to send and log. For auto/local, ask the server once (so switching
        models with ./model.sh needs no config change); falls back to the configured name."""
        if self.model not in AUTO_MODEL:
            return self.model
        if self._resolved is None:
            try:
                self._resolved = self._client.with_options(timeout=5).models.list().data[0].id
            except Exception:       # server down or an unusual server: try again next time
                return self.model
        return self._resolved

    def wait_ready(self, timeout: float = 900.0, poll: float = 2.0) -> bool:
        """Wait until the server can generate.

        llama-server answers /v1/models while it is still loading the model, so that is no
        proof it is ready; /health says 503 until it is. Servers without /health (not
        llama.cpp) are assumed ready. Returns False if the timeout runs out.
        """
        root = self.base_url.rstrip("/")
        url = (root[:-3] if root.endswith("/v1") else root) + "/health"
        start = time.monotonic()
        told = False
        while True:
            try:
                with _urlreq.urlopen(url, timeout=3):
                    return True
            except _urlerror.HTTPError as e:
                if e.code != 503:
                    return True
                state = "the model is still loading"
            except OSError:
                state = "can't reach the server yet"
            waited = time.monotonic() - start
            if waited >= timeout:
                self._console.note(f"[warning] gave up after {int(waited)}s: {state} ({url})")
                return False
            if not told:
                self._console.note(f"[waiting for the model server: {state}...]")
                told = True
            else:
                self._console.debug(0, f"[waiting for the model server: {state}, {int(waited)}s]")
            time.sleep(poll)

    @contextlib.contextmanager
    def _still_waiting(self, depth: int):
        """While a request is pending, say so every HEARTBEAT seconds (--debug), so a slow
        model looks different from a stuck program."""
        done = threading.Event()

        def beat():
            start = time.monotonic()
            while not done.wait(self.HEARTBEAT):
                self._console.debug(depth, f"[waiting for the model... {int(time.monotonic() - start)}s]")

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()

    def chat(self, messages: list, tools: list | None = None, *, depth: int = 0,
             label: str = "main", step: int | None = None):
        """One completion; returns the response, or None if every attempt failed.

        Local servers sometimes fail to parse a model's output, so this retries.
        """
        for attempt, temperature in enumerate(self.TEMPERATURES, start=1):
            try:
                kwargs = dict(model=self.model_name, messages=messages, temperature=temperature)
                if tools:
                    kwargs["tools"] = tools
                with self._still_waiting(depth):
                    return self._client.chat.completions.create(**kwargs)
            except self._errors as e:
                self._console.debug(depth, f"[llm error] attempt {attempt}/{len(self.TEMPERATURES)}: {e}")
                self._log.log("llm_error", agent=label, step=step, attempt=attempt, error=str(e))
        return None
