"""Context management, failure classification and retry behaviour. No server or `openai` needed."""
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import CONTEXT_FULL, FINISHED, LLM_FAILED, Agent  # noqa: E402
from agent.console import Console                                    # noqa: E402
from agent.context import NOTE_PREFIX, compact, message_chars, total_chars  # noqa: E402
from agent.llm import BAD_CALL_HINT, LLM, ContextFull               # noqa: E402
from agent.storage import EventLog                                   # noqa: E402
from agent.tools import FinishTool                                   # noqa: E402


# ---- history builders ------------------------------------------------------------------------
def call_msg(i, args=None, name="run_shell", reasoning=None):
    m = {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{i}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args or {"command": f"echo {i}"})}}]}
    if reasoning:
        m["reasoning_content"] = reasoning
    return m


def result_msg(i, text="ok"):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


def history(steps, output_chars=3000, goal="build it"):
    msgs = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": f"GOAL: {goal}"}]
    for i in range(steps):
        msgs += [call_msg(i, reasoning="thinking " * 100), result_msg(i, "line\n" * (output_chars // 5))]
    return msgs


def assert_valid(test, msgs):
    """Every tool call is answered, and no result is orphaned."""
    pending = set()
    for m in msgs:
        if m["role"] == "assistant":
            test.assertFalse(pending, "a tool call was left unanswered")
            pending = {tc["id"] for tc in m.get("tool_calls") or []}
        elif m["role"] == "tool":
            test.assertIn(m["tool_call_id"], pending, "orphaned tool result")
            pending.discard(m["tool_call_id"])
        else:
            test.assertFalse(pending, "a tool call was left unanswered")


class CompactTest(unittest.TestCase):
    def test_nothing_changes_when_it_already_fits(self):
        msgs = history(3, 200)
        before = json.dumps(msgs)
        rep = compact(msgs, total_chars(msgs) + 10)
        self.assertFalse(rep.changed)
        self.assertEqual(json.dumps(msgs), before)

    def test_old_reasoning_goes_first_and_recent_messages_are_untouched(self):
        msgs = history(8, 100)
        recent = json.dumps(msgs[-6:])
        compact(msgs, total_chars(msgs) - 3000)
        self.assertFalse(any("reasoning_content" in m for m in msgs[:-6]))
        self.assertEqual(json.dumps(msgs[-6:]), recent)

    def test_old_tool_outputs_are_cut_to_head_and_tail(self):
        msgs = history(8, 3000)
        newest = msgs[-1]["content"]
        rep = compact(msgs, total_chars(msgs) // 2)
        self.assertGreater(rep.trimmed_outputs, 0)
        old = [m for m in msgs[:-6] if m["role"] == "tool"]
        self.assertTrue(all("chars trimmed" in m["content"] and len(m["content"]) < 600 for m in old))
        self.assertEqual(msgs[-1]["content"], newest)           # the newest output is whole

    def test_long_arguments_of_old_calls_are_cut_and_stay_valid_json(self):
        big = "a" * 9000
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "GOAL: g"},
                call_msg(0, {"path": "x.js", "content": big}, "write_file"), result_msg(0, "wrote"),
                *sum(([call_msg(i), result_msg(i)] for i in range(1, 5)), [])]
        rep = compact(msgs, total_chars(msgs) - 8000)
        self.assertGreater(rep.trimmed_calls, 0)
        args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])     # still parses
        self.assertEqual(args["path"], "x.js")
        self.assertLess(len(args["content"]), 500)
        self.assertIn("trimmed", args["content"])

    def test_old_steps_are_dropped_oldest_first_with_one_note(self):
        msgs = history(12, 3000)
        rep = compact(msgs, 4000)
        self.assertGreater(rep.dropped_steps, 0)
        self.assertEqual(msgs[0]["content"], "SYSTEM")
        self.assertTrue(any(m["content"] == "GOAL: build it" for m in msgs if m["role"] == "user"))
        self.assertEqual(sum(str(m.get("content", "")).startswith(NOTE_PREFIX) for m in msgs), 1)
        self.assertEqual(msgs[-1]["tool_call_id"], "c11")                 # the newest step survived
        assert_valid(self, msgs)

    def test_repeated_compaction_keeps_a_single_note_and_valid_history(self):
        msgs = history(10, 3000)
        compact(msgs, 5000)
        msgs += [call_msg(50), result_msg(50, "z" * 3000), call_msg(51), result_msg(51, "z" * 3000)]
        compact(msgs, 3500)
        self.assertEqual(sum(str(m.get("content", "")).startswith(NOTE_PREFIX) for m in msgs), 1)
        assert_valid(self, msgs)
        self.assertEqual(msgs[0]["role"], "system")

    def test_when_the_recent_messages_are_the_bulky_ones_only_the_newest_step_is_protected(self):
        msgs = history(4, 3000)                           # the last 6 messages are 3 big steps
        newest = json.dumps(msgs[-2:])
        rep = compact(msgs, 6500)                         # fits only if steps 1 and 2 shrink too
        self.assertTrue(rep.changed)
        self.assertLessEqual(total_chars(msgs), 6500)
        self.assertEqual(json.dumps(msgs[-2:]), newest)   # the newest call and result are whole
        trimmed = [m["tool_call_id"] for m in msgs if m["role"] == "tool" and "chars trimmed" in m["content"]]
        self.assertIn("c1", trimmed)                      # these were inside the usual protected window
        self.assertNotIn("c3", trimmed)
        assert_valid(self, msgs)

    def test_the_current_goal_survives_even_when_everything_else_goes(self):
        msgs = history(10, 3000, goal="the important one")
        compact(msgs, 100)                                                # impossible target
        self.assertTrue(any(m.get("content") == "GOAL: the important one" for m in msgs))
        assert_valid(self, msgs)

    def test_older_goals_can_be_dropped_but_the_latest_stays(self):
        msgs = history(4, 2000, goal="first") + history(4, 2000, goal="second")[1:]
        compact(msgs, 3000)
        goals = [m["content"] for m in msgs if str(m.get("content", "")).startswith("GOAL:")]
        self.assertEqual(goals[-1], "GOAL: second")
        assert_valid(self, msgs)

    def test_size_accounts_for_the_tool_schemas_sent_alongside(self):
        msgs = history(2, 100)
        self.assertEqual(total_chars(msgs, 500), total_chars(msgs) + 500)
        self.assertGreater(message_chars(call_msg(0, {"command": "x" * 1000})), 1000)
        self.assertGreater(message_chars(call_msg(0, reasoning="r" * 500)), message_chars(call_msg(0)) + 499)


# ---- the agent loop ----------------------------------------------------------------------------
class Reply:
    """A model reply that calls `finish`."""
    def __init__(self, answer="done"):
        self.content = None
        self.tool_calls = [NS(id="f1", function=NS(name="finish", arguments=json.dumps({"answer": answer})))]

    def model_dump(self, exclude_none=False):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": "f1", "type": "function", "function": {
                    "name": "finish", "arguments": self.tool_calls[0].function.arguments}}]}


class ScriptedLLM:
    def __init__(self, *script, n_ctx=None, failure="the model server kept failing"):
        self.script, self.calls, self.n_ctx, self.failure = list(script), [], n_ctx, failure

    def context_size(self):
        return self.n_ctx

    def failure_text(self):
        return self.failure

    def chat(self, messages, tools=None, *, depth=0, label="main", step=None, raise_context_full=False):
        self.calls.append(json.loads(json.dumps(messages)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return None if item is None else NS(choices=[NS(message=item)])


class AgentContextTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = EventLog(f"{self._tmp.name}/log.jsonl")

    def agent(self, llm):
        return Agent("main", llm, [FinishTool()], 5, self.log, Console())

    def events(self, name):
        return [e for e in map(json.loads, Path(self.log.path).read_text().splitlines()) if e["event"] == name]

    def test_makes_room_before_asking_when_the_history_is_close_to_the_window(self):
        llm = ScriptedLLM(Reply(), n_ctx=4000)
        msgs = history(10, 3000)                       # ~30k chars = ~10k tokens in a 4k window
        result = self.agent(llm).run(msgs)
        self.assertEqual(result.status, FINISHED)
        sent = llm.calls[0]
        self.assertLess(total_chars(sent), 8000)       # shrunk before the very first request
        self.assertEqual(sent[0]["content"], "SYSTEM")
        assert_valid(self, sent)
        ev = self.events("compact")
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["reason"], "nearly full")
        self.assertLess(ev[0]["after_tokens"], ev[0]["before_tokens"])

    def test_leaves_the_history_alone_when_it_fits(self):
        llm = ScriptedLLM(Reply(), n_ctx=100_000)
        msgs = history(3, 500)
        before = json.dumps(msgs)
        self.agent(llm).run(msgs)
        self.assertEqual(json.dumps(llm.calls[0]), before)
        self.assertEqual(self.events("compact"), [])

    def test_unknown_window_means_no_proactive_compaction(self):
        llm = ScriptedLLM(Reply(), n_ctx=None)
        self.agent(llm).run(history(10, 3000))
        self.assertEqual(self.events("compact"), [])

    def test_when_the_server_says_full_it_shrinks_and_retries_once(self):
        llm = ScriptedLLM(ContextFull(20000, 16384), Reply("recovered"), n_ctx=None)
        msgs = history(10, 3000)
        result = self.agent(llm).run(msgs)
        self.assertEqual((result.status, result.text), (FINISHED, "recovered"))
        self.assertEqual(len(llm.calls), 2)
        self.assertLess(total_chars(llm.calls[1]), total_chars(llm.calls[0]) // 2)
        assert_valid(self, llm.calls[1])
        self.assertEqual(self.events("compact")[0]["reason"], "the server said it was full")

    def test_gives_up_with_an_explanation_if_it_is_still_full(self):
        why = "the conversation no longer fits in the model's context window (...)"
        llm = ScriptedLLM(ContextFull(20000, 16384), ContextFull(20000, 16384), failure=why)
        result = self.agent(llm).run(history(10, 3000))
        self.assertEqual((result.status, result.text), (CONTEXT_FULL, why))
        self.assertEqual(len(llm.calls), 2)            # one retry, not an endless loop

    def test_gives_up_when_nothing_can_be_shrunk(self):
        llm = ScriptedLLM(ContextFull(20000, 16384), ContextFull(20000, 16384))
        tiny = [{"role": "system", "content": "S"}, {"role": "user", "content": "GOAL: g"}]
        result = self.agent(llm).run(tiny)
        self.assertEqual(result.status, CONTEXT_FULL)
        self.assertEqual(len(llm.calls), 1)            # nothing to shrink: no pointless retry

    def test_an_ordinary_failure_is_reported_with_the_servers_explanation(self):
        llm = ScriptedLLM(None, failure="can't reach the model server at http://x/v1.")
        result = self.agent(llm).run(history(1, 100))
        self.assertEqual((result.status, result.text), (LLM_FAILED, "can't reach the model server at http://x/v1."))

    def test_token_estimate_learns_from_the_servers_count(self):
        a = self.agent(ScriptedLLM())
        a._calibrate(NS(usage=NS(prompt_tokens=1000)), 4000)
        self.assertAlmostEqual(a._tokens_per_char, 0.25)
        a._calibrate(NS(usage=None), 4000)               # no usage info: unchanged
        a._calibrate(NS(usage=NS(prompt_tokens=1)), 4000000)   # absurd: clamped
        self.assertGreaterEqual(a._tokens_per_char, 0.15)


# ---- LLM failure classification and retries -----------------------------------------------------
class StatusErr(Exception):
    def __init__(self, text, status_code=500):
        super().__init__(text)
        self.status_code = status_code


class ConnErr(Exception):
    pass


CTX_400 = ("Error code: 400 - {'error': {'code': 400, 'message': 'request (16726 tokens) exceeds the available "
           "context size (16384 tokens), try increasing it', 'type': 'exceed_context_size_error'}}")
PARSE_500 = ("Error code: 500 - {'error': {'code': 500, 'message': 'Failed to parse tool call arguments as JSON: "
             "[json.exception.parse_error.101] missing closing quote', 'type': 'server_error'}}")


class LLMFailureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def make(self, *results, n_ctx=None):
        """An LLM whose server returns `results` in order (exceptions are raised)."""
        self.requests = []
        queue = list(results)

        def create(**kw):
            self.requests.append(json.loads(json.dumps(kw["messages"])))
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        stub = types.ModuleType("openai")
        stub.APIStatusError, stub.APIConnectionError = StatusErr, ConnErr
        stub.OpenAI = lambda **kw: NS(chat=NS(completions=NS(create=create)))
        with mock.patch.dict(sys.modules, {"openai": stub}):
            return LLM("http://x:8080/v1", "m", EventLog(f"{self._tmp.name}/l.jsonl"), Console(), n_ctx=n_ctx)

    MSGS = [{"role": "user", "content": "hi"}]

    def test_context_overflow_is_not_retried_and_explains_itself(self):
        llm = self.make(StatusErr(CTX_400, 400))
        self.assertIsNone(llm.chat(self.MSGS))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(llm.last_failure.kind, "context")
        self.assertEqual(llm.last_failure.detail, "16726 of 16384 tokens")
        self.assertIn("--new", llm.failure_text())
        self.assertIn("bigger window", llm.failure_text())

    def test_context_overflow_raises_when_the_caller_can_handle_it(self):
        llm = self.make(StatusErr(CTX_400, 400))
        with self.assertRaises(ContextFull) as cm:
            llm.chat(self.MSGS, raise_context_full=True)
        self.assertEqual((cm.exception.n_prompt, cm.exception.n_ctx), (16726, 16384))
        self.assertEqual(llm._n_ctx, 16384)                     # learned the window from the error

    def test_a_cut_off_tool_call_is_retried_with_a_hint_for_shorter_calls(self):
        ok = NS(choices=[], usage=NS(prompt_tokens=500))
        llm = self.make(StatusErr(PARSE_500), StatusErr(PARSE_500), ok)
        self.assertIs(llm.chat(self.MSGS), ok)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.requests[0], self.MSGS)                           # first try: as is
        for retry in self.requests[1:]:
            self.assertEqual(retry[-1], {"role": "user", "content": BAD_CALL_HINT})
            self.assertEqual(retry[:-1], self.MSGS)
        self.assertEqual(self.MSGS, [{"role": "user", "content": "hi"}])        # caller's list untouched
        self.assertIsNone(llm.last_failure)
        self.assertEqual(llm.last_prompt_tokens, 500)

    def test_a_cut_off_tool_call_with_a_nearly_full_window_is_a_context_problem(self):
        ok = NS(choices=[], usage=NS(prompt_tokens=15664))
        llm = self.make(ok, StatusErr(PARSE_500), n_ctx=16384)
        llm.chat(self.MSGS)                                                     # previous step: 15664 of 16384
        with self.assertRaises(ContextFull):
            llm.chat(self.MSGS, raise_context_full=True)
        self.assertEqual(len(self.requests), 2)                                 # no hint-retries: they cannot help
        self.assertEqual(llm.last_failure.kind, "context")

    def test_persistent_parse_failures_are_explained(self):
        llm = self.make(*[StatusErr(PARSE_500)] * 3)
        self.assertIsNone(llm.chat(self.MSGS))
        self.assertEqual(llm.last_failure.kind, "bad_tool_call")
        self.assertIn("could not parse", llm.failure_text())

    def test_unreachable_server_is_named_as_such(self):
        llm = self.make(*[ConnErr("Connection refused")] * 3)
        self.assertIsNone(llm.chat(self.MSGS))
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(llm.last_failure.kind, "unreachable")
        self.assertIn("can't reach the model server at http://x:8080/v1", llm.failure_text())
        self.assertNotIn(BAD_CALL_HINT, json.dumps(self.requests))               # the hint is only for bad calls

    def test_other_errors_keep_the_old_behaviour_and_say_what_happened(self):
        llm = self.make(*[StatusErr("Error code: 503 - busy", 503)] * 3)
        self.assertIsNone(llm.chat(self.MSGS))
        self.assertEqual(len(self.requests), 3)
        self.assertTrue(llm.failure_text().startswith("the model server kept failing"))
        self.assertIn("503", llm.failure_text())

    def test_context_size_comes_from_props_once(self):
        llm = self.make()
        props = io.BytesIO(b'{"default_generation_settings": {"n_ctx": 16384}}')
        with mock.patch("agent.llm._urlreq.urlopen", return_value=props) as op:
            self.assertEqual((llm.context_size(), llm.context_size()), (16384, 16384))
        self.assertEqual(op.call_count, 1)
        self.assertEqual(op.call_args[0][0], "http://x:8080/props")             # /v1 stripped

    def test_unknown_window_is_none_and_not_asked_again(self):
        llm = self.make()
        with mock.patch("agent.llm._urlreq.urlopen", side_effect=OSError("no /props")) as op:
            self.assertEqual((llm.context_size(), llm.context_size()), (None, None))
        self.assertEqual(op.call_count, 1)

    def test_a_configured_window_wins_and_needs_no_request(self):
        llm = self.make(n_ctx=8192)
        with mock.patch("agent.llm._urlreq.urlopen") as op:
            self.assertEqual(llm.context_size(), 8192)
        op.assert_not_called()


if __name__ == "__main__":
    unittest.main()
