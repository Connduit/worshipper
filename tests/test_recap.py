"""The reminder shown when a saved conversation is resumed."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.app import App                                   # noqa: E402
from agent.config import Config                             # noqa: E402
from agent.context import NOTE                              # noqa: E402
from agent.recap import exchanges, format_recap             # noqa: E402
from agent.util import fmt_duration                         # noqa: E402


def goal(text):
    return {"role": "user", "content": f"GOAL: {text}"}


def finish(answer, i=0):
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"f{i}", "type": "function",
                 "function": {"name": "finish", "arguments": json.dumps({"answer": answer})}}]},
            {"role": "tool", "tool_call_id": f"f{i}", "content": "Goal marked complete. Wait for the next goal."}]


def step(i):
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "files"}]


SYSTEM = {"role": "system", "content": "S"}


class ExchangesTest(unittest.TestCase):
    def test_goals_are_paired_with_their_final_answers(self):
        msgs = [SYSTEM, goal("one"), *step(1), *finish("did one", 1), goal("two"), *step(2), *finish("did two", 2)]
        self.assertEqual([(e.goal, e.answer) for e in exchanges(msgs)], [("one", "did one"), ("two", "did two")])

    def test_a_run_that_never_finished_has_no_answer(self):
        msgs = [SYSTEM, goal("one"), *finish("did one"), goal("two"), *step(2)]       # interrupted during goal two
        self.assertEqual([e.answer for e in exchanges(msgs)], ["did one", None])

    def test_the_last_finish_wins_and_unreadable_ones_are_ignored(self):
        bad = [{"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "finish", "arguments": "{not json"}}]}]
        self.assertEqual(exchanges([SYSTEM, goal("g"), *bad])[0].answer, None)
        self.assertEqual(exchanges([SYSTEM, goal("g"), *finish("first"), *finish("second", 1)])[0].answer, "second")

    def test_other_messages_are_not_goals(self):
        msgs = [SYSTEM, {"role": "user", "content": "just chatting"}, {"role": "user", "content": NOTE}, goal("real")]
        self.assertEqual([e.goal for e in exchanges(msgs)], ["real"])

    def test_empty_history(self):
        self.assertEqual(exchanges([SYSTEM]), [])


class FormatTest(unittest.TestCase):
    def history(self, n):
        msgs = [SYSTEM]
        for i in range(1, n + 1):
            msgs += [goal(f"goal number {i}"), *step(i), *finish(f"answer {i}", i)]
        return msgs

    def test_shows_the_last_three_goals_with_their_numbers_and_the_total(self):
        text = format_recap(self.history(5), 2 * 3600 + 900)
        self.assertIn("last used 2h 15m ago, 5 goals so far", text)
        self.assertIn("The last 3:", text)
        for n in (3, 4, 5):
            self.assertIn(f"{n}. you:   goal number {n}", text)
            self.assertIn(f"agent: answer {n}", text)
        self.assertNotIn("goal number 2", text)

    def test_a_single_goal_reads_naturally(self):
        text = format_recap(self.history(1), 45)
        self.assertIn("(last used 45s ago, 1 goal so far). The last one:", text)

    def test_unknown_age_is_left_out(self):
        self.assertIn("(2 goals so far)", format_recap(self.history(2), None))

    def test_long_and_multiline_text_is_flattened_and_cut(self):
        msgs = [SYSTEM, goal("build\nthe   thing " + "x" * 300), *finish("line one\nline two " + "y" * 300)]
        lines = format_recap(msgs, 5, width=60).splitlines()
        you, agent = lines[1], lines[2]
        self.assertTrue(you.endswith("...") and agent.endswith("..."))
        self.assertIn("build the thing", you)
        self.assertIn("line one line two", agent)
        self.assertLessEqual(len(you.split("you:   ")[1]), 60)
        self.assertLessEqual(len(agent.split("agent: ")[1]), 60)

    def test_an_unfinished_goal_says_so(self):
        msgs = [SYSTEM, goal("first"), *finish("ok"), goal("the one that was cut short"), *step(9)]
        text = format_recap(msgs, 60)
        self.assertIn("the one that was cut short", text)
        self.assertIn("no final answer", text)

    def test_mentions_trimmed_history(self):
        msgs = [SYSTEM, {"role": "user", "content": NOTE}, goal("g"), *finish("a")]
        self.assertIn("older steps were trimmed", format_recap(msgs, 1))
        self.assertNotIn("trimmed", format_recap([SYSTEM, goal("g"), *finish("a")], 1))

    def test_nothing_to_recap(self):
        self.assertEqual(format_recap([SYSTEM], 5), "")
        self.assertEqual(format_recap([SYSTEM, {"role": "user", "content": "hi"}], 5), "")

    def test_durations_cover_days(self):
        self.assertEqual([fmt_duration(s) for s in (45, 72, 3725, 183000)], ["45s", "1m 12s", "1h 02m", "2d 02h"])


class AppRecapTest(unittest.TestCase):
    """The reminder appears once, when the user comes back, and not for a fresh conversation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.session = f"{self.tmp}/session.json"

    def env(self, **extra):
        return {"WORKSPACE": f"{self.tmp}/ws", "SESSION_FILE": self.session, "LOG_FILE": f"{self.tmp}/log.jsonl",
                "FRIEND_MEMORY_FILE": f"{self.tmp}/mem.json", "SANDBOX_MODE": "none", "LLM_URL": "http://x/v1", **extra}

    def save_history(self, age_seconds=7200):
        Path(self.session).write_text(json.dumps(
            [SYSTEM, goal("make a car part picker"), *step(1), *finish("built index.html with search", 1)]))
        old = Path(self.session).stat().st_mtime - age_seconds
        os.utime(self.session, (old, old))

    def app(self, **extra):
        class Llm:                               # a server that is up but never answers usefully
            wait_ready = staticmethod(lambda *a, **k: True)
            model_name = "fake"
            context_size = staticmethod(lambda: None)
            failure_text = staticmethod(lambda: "no model here")
            chat = staticmethod(lambda *a, **k: None)
        return App(Config.from_env(self.env(**extra)), llm=Llm())

    def test_chat_shows_the_reminder_once_before_the_first_prompt(self):
        self.save_history(7200)
        app = self.app()
        app.setup()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("builtins.input", side_effect=EOFError):
            app.chat()
            first = out.getvalue()
            app.chat()                           # a second visit in the same process: no repeat
        self.assertIn("Resuming your previous conversation (last used 2h 00m ago, 1 goal so far)", first)
        self.assertIn("you:   make a car part picker", first)
        self.assertIn("agent: built index.html with search", first)
        self.assertEqual(out.getvalue(), first + "\n")        # only the newline printed on EOF

    def test_a_fresh_conversation_shows_nothing(self):
        app = self.app()
        app.setup(new=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("builtins.input", side_effect=EOFError):
            app.chat()
        self.assertNotIn("Resuming", out.getvalue())

    def test_history_without_goals_shows_nothing(self):
        Path(self.session).write_text(json.dumps([SYSTEM]))
        app = self.app()
        app.setup()
        self.assertEqual(app._recap, "")

    def test_a_one_shot_goal_shows_it_only_with_debug(self):
        for debug, expect in (("0", False), ("1", True)):
            self.save_history()
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                app = self.app(DEBUG=debug)
                app.setup()
                app.run_goal("next")
            self.assertEqual("Resuming your previous conversation" in out.getvalue(), expect, f"DEBUG={debug}")

    def test_a_goal_run_before_chat_suppresses_the_reminder(self):
        self.save_history()
        app = self.app()
        app.setup()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            app.run_goal("next")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch("builtins.input", side_effect=EOFError):
            app.chat()
        self.assertNotIn("Resuming", out.getvalue())


if __name__ == "__main__":
    unittest.main()
