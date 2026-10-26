"""Talking to the llama.cpp server (OpenAI-compatible)."""
from __future__ import annotations

from .console import Console
from .storage import EventLog


AUTO_MODEL = ("", "auto", "local")   # "ask the server which model it is serving"


class LLM:
    """One chat-completions endpoint + model, with retries."""

    TEMPERATURES = (0.2, 0.7, 1.0)    # one per attempt: vary the sample on retries

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
                return self._client.chat.completions.create(**kwargs)
            except self._errors as e:
                self._console.debug(depth, f"[llm error] attempt {attempt}/{len(self.TEMPERATURES)}: {e}")
                self._log.log("llm_error", agent=label, step=step, attempt=attempt, error=str(e))
        return None
