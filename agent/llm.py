"""Talking to the llama.cpp server (OpenAI-compatible)."""
from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from dataclasses import dataclass
from urllib import error as _urlerror, request as _urlreq

from .console import Console
from .storage import EventLog
from .util import fmt_duration


AUTO_MODEL = ("", "auto", "local")   # "ask the server which model it is serving"

# What llama-server says when the request does not fit its context window.
_CONTEXT_RE = re.compile(r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)")

BAD_CALL_HINT = ("Your last tool call was cut off or was not valid JSON, so it could not be run. "
                 "Call a tool again with SHORTER arguments: create files with write_file in pieces of "
                 "about 100 lines (use append=true for the following pieces), and avoid heredocs in run_shell.")


class ContextFull(Exception):
    """The conversation no longer fits in the model's context window. Retrying cannot help;
    the history has to shrink (see context.compact)."""

    def __init__(self, n_prompt: int | None = None, n_ctx: int | None = None):
        super().__init__(f"{n_prompt} tokens do not fit in a context of {n_ctx}")
        self.n_prompt, self.n_ctx = n_prompt, n_ctx


@dataclass
class Failure:
    """Why the last chat() gave up. kind: context | bad_tool_call | unreachable | other"""
    kind: str
    detail: str = ""


class LLM:
    """One chat-completions endpoint + model, with retries."""

    TEMPERATURES = (0.2, 0.7, 1.0)    # one per attempt: vary the sample on retries
    TICK = 1.0                        # redraw the live status line this often (terminals)
    HEARTBEAT = 15.0                  # seconds between "still waiting" log lines (--debug, no terminal)

    def __init__(self, base_url: str, model: str, log: EventLog, console: Console,
                 api_key: str = "not-needed-for-local", n_ctx: int | None = None):
        # Imported here so -h, --diff and --friends work without `openai` installed.
        from openai import APIConnectionError, APIStatusError, OpenAI
        self.base_url = base_url
        self.model = model
        self._log = log
        self._console = console
        self._errors = (APIStatusError, APIConnectionError)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._resolved: str | None = None
        self._n_ctx = n_ctx                 # context window per request; None = ask the server
        self._props_tried = n_ctx is not None
        self.last_prompt_tokens: int | None = None
        self.last_failure: Failure | None = None

    def _root(self) -> str:
        root = self.base_url.rstrip("/")
        return root[:-3] if root.endswith("/v1") else root

    def context_size(self) -> int | None:
        """The context window a request must fit in, from llama-server's /props (asked once).
        None if unknown; the agent then only reacts to the server's own 'too big' error."""
        if self._n_ctx is None and not self._props_tried:
            self._props_tried = True
            try:
                with _urlreq.urlopen(self._root() + "/props", timeout=3) as r:
                    self._n_ctx = int(json.load(r)["default_generation_settings"]["n_ctx"])
            except Exception:
                pass
        return self._n_ctx

    def failure_text(self) -> str:
        """The last failure in plain words, with what to do about it."""
        f = self.last_failure
        if f is None or f.kind == "other":
            short = (f.detail[:160] + "...") if f and len(f.detail) > 160 else (f.detail if f else "")
            return "the model server kept failing" + (f" ({short})" if short else "")
        if f.kind == "context":
            return ("the conversation no longer fits in the model's context window"
                    f" ({f.detail}). Start fresh with --new, or give the server a bigger window "
                    "(for example CTX=32768 ./model.sh <model>).")
        if f.kind == "bad_tool_call":
            return ("the model kept producing tool calls the server could not parse (usually a reply cut "
                    "off half-way). Try a smaller step, --new, or a bigger context window.")
        return f"can't reach the model server at {self.base_url}. Is it running? (./model.sh status)"

    def _classify(self, e: Exception) -> str:
        text = str(e)
        if "exceeds the available context size" in text or "exceed_context_size_error" in text:
            return "context"
        if "Failed to parse tool call" in text or "does not match the expected" in text:
            return "bad_tool_call"
        if getattr(e, "status_code", None) is None and isinstance(e, self._errors[1]):
            return "unreachable"
        return "other"

    def _near_limit(self) -> bool:
        return bool(self._n_ctx and self.last_prompt_tokens and self.last_prompt_tokens >= 0.85 * self._n_ctx)

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
        url = self._root() + "/health"
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
    def _still_waiting(self, depth: int, label: str = "main", step: int | None = None):
        """While a request is pending, show a live timer so a slow model looks different from
        a hung program: an in-place status line on a terminal, or (with --debug and no
        terminal) a log line every HEARTBEAT seconds."""
        done = threading.Event()
        where = label if step is None else f"{label}, step {step}"

        def beat():
            start = last_line = time.monotonic()
            while not done.wait(self.TICK):
                now = time.monotonic()
                shown = self._console.status(f"{where}: waiting for the model... {fmt_duration(now - start)}")
                if not shown and now - last_line >= self.HEARTBEAT:
                    self._console.debug(depth, f"[waiting for the model... {fmt_duration(now - start)}]")
                    last_line = now

        thread = threading.Thread(target=beat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join(timeout=2)
            self._console.clear_status()

    def chat(self, messages: list, tools: list | None = None, *, depth: int = 0,
             label: str = "main", step: int | None = None, raise_context_full: bool = False):
        """One completion; returns the response, or None if every attempt failed (then
        last_failure / failure_text() say why).

        Failures are told apart, because they need different handling:
          - the conversation does not fit the window: retrying cannot help. Raises ContextFull
            if asked to (the agent shrinks the history), else gives up at once.
          - a tool call the server could not parse: usually the reply was cut off at the end of
            the window or ran too long. If the window was nearly full that is a context problem;
            otherwise retry with a hint to write shorter calls.
          - anything else (server busy, unreachable...): retry with a different sample.
        """
        failure = None
        total = len(self.TEMPERATURES)
        for attempt, temperature in enumerate(self.TEMPERATURES, start=1):
            sent = messages
            if failure is not None and failure.kind == "bad_tool_call":
                sent = messages + [{"role": "user", "content": BAD_CALL_HINT}]   # for this request only
            try:
                kwargs = dict(model=self.model_name, messages=sent, temperature=temperature)
                if tools:
                    kwargs["tools"] = tools
                with self._still_waiting(depth, label, step):
                    resp = self._client.chat.completions.create(**kwargs)
                usage = getattr(resp, "usage", None)
                self.last_prompt_tokens = getattr(usage, "prompt_tokens", None) or self.last_prompt_tokens
                self.last_failure = None
                return resp
            except self._errors as e:
                kind = self._classify(e)
                if kind == "bad_tool_call" and self._near_limit():
                    kind = "context"        # cut off because the window was nearly full
                self._log.log("llm_error", agent=label, step=step, attempt=attempt, kind=kind, error=str(e))
                if kind == "context":
                    m = _CONTEXT_RE.search(str(e))
                    if m:
                        self.last_prompt_tokens, self._n_ctx = int(m.group(1)), int(m.group(2))
                    detail = (f"{self.last_prompt_tokens} of {self._n_ctx} tokens"
                              if self.last_prompt_tokens and self._n_ctx else "it is full")
                    self.last_failure = Failure("context", detail)
                    self._console.debug(depth, f"[llm error] the conversation does not fit the context window "
                                               f"({detail}); retrying cannot help")
                    if raise_context_full:
                        raise ContextFull(self.last_prompt_tokens, self._n_ctx) from None
                    return None
                failure = Failure(kind, str(e))
                why = {"bad_tool_call": "the model's tool call was cut off or malformed",
                       "unreachable": "can't reach the server"}.get(kind, "the server returned an error")
                nxt = "retrying" + (" with a hint to write shorter calls" if kind == "bad_tool_call" else "")
                self._console.debug(depth, f"[llm error] attempt {attempt}/{total}: {why}"
                                           f"{'; ' + nxt if attempt < total else ''}\n    {e}")
        self.last_failure = failure
        return None
