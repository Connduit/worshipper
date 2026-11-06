"""What the status line says about the current step, and the live timers behind it."""
import contextlib
import functools
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import FINISHED, Agent                    # noqa: E402
from agent.console import Console, working                 # noqa: E402
from agent.context import NOTE                             # noqa: E402
from agent.llm import LLM                                  # noqa: E402
from agent.stepinfo import reacting_to, running, short     # noqa: E402
from agent.storage import EventLog                         # noqa: E402
from agent.tools import CallContext, FinishTool, Tool      # noqa: E402


def call(i, name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def result(i, text):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


GOAL = {"role": "user", "content": "GOAL: make me a bmw part picker page"}


class ShortTest(unittest.TestCase):
    def test_flattens_whitespace_and_cuts_with_dots(self):
        self.assertEqual(short("a\n  b\tc", 20), "a b c")
        self.assertEqual(short("x" * 50, 10), "xxxxxxx...")
        self.assertEqual(short("exact", 5), "exact")


class ReactingToTest(unittest.TestCase):
    def test_first_step_names_the_goal(self):
        self.assertEqual(reacting_to([{"role": "system", "content": "S"}, GOAL]),
                         'starting on "make me a bmw part picker page"')

    def test_long_goals_are_shortened(self):
        text = reacting_to([{"role": "user", "content": "GOAL: " + "word " * 40}])
        self.assertTrue(text.endswith('..."'))
        self.assertLess(len(text), 70)

    def test_after_a_command_it_says_which_and_how_it_went(self):
        msgs = [GOAL, call(1, "run_shell", command="npm   test"), result(1, "exit code: 1\nstdout:\nfail")]
        self.assertEqual(reacting_to(msgs), "reacting to `npm test` (exit 1)")
        msgs[-1] = result(1, "exit code: 0\nstdout:\nok")
        self.assertEqual(reacting_to(msgs), "reacting to `npm test` (exit 0)")

    def test_long_commands_are_shortened(self):
        msgs = [GOAL, call(1, "run_shell", command="echo " + "a" * 200), result(1, "exit code: 0")]
        self.assertLess(len(reacting_to(msgs)), 80)

    def test_files_and_failures(self):
        msgs = [GOAL, call(1, "write_file", path="src/main.js", content="x"), result(1, "wrote 1 bytes")]
        self.assertEqual(reacting_to(msgs), "reacting to write_file src/main.js")
        msgs = [GOAL, call(1, "write_file", path="a.txt", content="x", append=True), result(1, "ERROR: outside")]
        self.assertEqual(reacting_to(msgs), "reacting to write_file a.txt (append) (failed)")
        msgs = [GOAL, call(1, "run_shell", command="sleep 99"), result(1, "ERROR: command timed out after 30s")]
        self.assertEqual(reacting_to(msgs), "reacting to `sleep 99` (timed out)")

    def test_with_several_results_it_matches_the_last_one_to_its_own_call(self):
        two = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "run_shell", "arguments": '{"command": "first"}'}},
            {"id": "b", "type": "function", "function": {"name": "run_shell", "arguments": '{"command": "second"}'}}]}
        msgs = [GOAL, two, {"role": "tool", "tool_call_id": "a", "content": "exit code: 0"},
                {"role": "tool", "tool_call_id": "b", "content": "exit code: 2"}]
        self.assertEqual(reacting_to(msgs), "reacting to `second` (exit 2)")

    def test_friends_and_other_tools(self):
        msgs = [GOAL, call(1, "ask_friend", friend="reviewer", question="?"), result(1, "looks fine")]
        self.assertEqual(reacting_to(msgs), "reacting to ask_friend reviewer")
        self.assertEqual(reacting_to([GOAL, call(1, "get_time"), result(1, "noon")]), "reacting to get_time")

    def test_after_context_trimming_and_for_odd_histories(self):
        self.assertEqual(reacting_to([GOAL, {"role": "user", "content": NOTE}]), "continuing after trimming old steps")
        self.assertEqual(reacting_to([GOAL, {"role": "user", "content": "hello"}]), "your last message")
        self.assertEqual(reacting_to([GOAL, result(9, "orphan")]), "reacting to a tool result")
        self.assertEqual(reacting_to([]), "")
        bad = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_shell", "arguments": "{not json"}}]}
        self.assertEqual(reacting_to([GOAL, bad, result(1, "exit code: 0")]), "reacting to `` (exit 0)")


class RunningTest(unittest.TestCase):
    def test_what_a_running_tool_is_doing(self):
        self.assertEqual(running("run_shell", {"command": "npm test"}), "running `npm test`")
        self.assertEqual(running("write_file", {"path": "a/b.js"}), "writing a/b.js")
        self.assertEqual(running("tune_server", {}), "restarting the model server")
        self.assertEqual(running("promote_changes", {}), "promoting your changes")
        self.assertEqual(running("mystery", {}), "running mystery")


class WorkingTest(unittest.TestCase):
    def test_a_terminal_gets_a_live_line_that_is_wiped_afterwards(self):
        console = mock.Mock()
        console.status.return_value = True
        with working(console, "main, step 2: running `ls`...", tick=0.02):
            time.sleep(0.12)
        text = console.status.call_args[0][0]
        self.assertTrue(text.startswith("main, step 2: running `ls`... "))
        self.assertTrue(text.endswith("s"))
        console.debug.assert_not_called()
        console.clear_status.assert_called()

    def test_without_a_terminal_it_logs_heartbeat_lines_with_the_text(self):
        console = mock.Mock()
        console.status.return_value = False
        with working(console, "main, step 2: running `ls`...", 1, tick=0.02, heartbeat=0.02):
            time.sleep(0.12)
        depth, line = console.debug.call_args[0]
        self.assertEqual(depth, 1)
        self.assertIn("main, step 2: running `ls`...", line)


class LLMContextTest(unittest.TestCase):
    def make(self, results, console):
        queue = list(results)
        self.sent = []

        def create(**kw):
            self.sent.append(kw)
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        class StatusErr(Exception):
            status_code = 500

        self.StatusErr = StatusErr
        stub = types.ModuleType("openai")
        stub.APIStatusError, stub.APIConnectionError = StatusErr, ConnectionError
        stub.OpenAI = lambda **kw: NS(chat=NS(completions=NS(create=create)))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        with mock.patch.dict(sys.modules, {"openai": stub}):
            llm = LLM("http://x/v1", "m", EventLog(f"{self._tmp.name}/l.jsonl"), console)
        llm.TICK = llm.HEARTBEAT = 0.02
        return llm

    def statuses(self, console):
        return [c[0][0] for c in console.status.call_args_list]

    def test_the_wait_says_what_the_step_is_for(self):
        console = mock.Mock()
        console.status.return_value = True
        slow = NS(choices=[])
        llm = self.make([slow], console)
        llm._client.chat.completions.create = lambda **kw: (time.sleep(0.12), slow)[1]
        llm.chat([{"role": "user", "content": "hi"}], label="main", step=3, doing="reacting to `npm test` (exit 1)")
        self.assertTrue(any(s.startswith("main, step 3: reacting to `npm test` (exit 1), waiting for the model...")
                            for s in self.statuses(console)), self.statuses(console))

    def test_a_retry_says_why(self):
        console = mock.Mock()
        console.status.return_value = True
        llm = self.make([None, NS(choices=[])], console)
        llm.TICK = 0.02
        ok = NS(choices=[])
        err = self.StatusErr("Failed to parse tool call arguments as JSON: bad")
        outcomes = iter([err, ok])

        def create(**kw):
            time.sleep(0.08)
            item = next(outcomes)
            if isinstance(item, Exception):
                raise item
            return item

        llm._client.chat.completions.create = create
        llm.chat([{"role": "user", "content": "hi"}], step=2, doing="starting on \"x\"")
        retry = [s for s in self.statuses(console) if "retry 2/3" in s]
        self.assertTrue(retry, self.statuses(console))
        self.assertIn("after a cut-off tool call", retry[0])
        self.assertIn('starting on "x"', retry[0])

    def test_no_context_keeps_the_plain_wording(self):
        console = mock.Mock()
        console.status.return_value = True
        llm = self.make([], console)
        llm._client.chat.completions.create = lambda **kw: (time.sleep(0.1), NS(choices=[]))[1]
        llm.chat([{"role": "user", "content": "hi"}], step=1)
        self.assertTrue(any(s.startswith("main, step 1: waiting for the model...") for s in self.statuses(console)))


class Reply:
    def __init__(self, name, **args):
        self.content = None
        self.tool_calls = [NS(id=f"r{id(self) % 9999}", function=NS(name=name, arguments=json.dumps(args)))]

    def model_dump(self, exclude_none=False):
        tc = self.tool_calls[0]
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]}


class Recorder:
    def __init__(self, *script):
        self.script, self.doing = list(script), []

    def context_size(self):
        return None

    def failure_text(self):
        return "failed"

    def chat(self, messages, tools=None, *, doing="", **kw):
        self.doing.append(doing)
        return NS(choices=[NS(message=self.script.pop(0))])


class SlowShell(Tool):
    name = "run_shell"
    description = "pretend shell"
    parameters = {"type": "object", "properties": {"command": {"type": "string"}}}
    delegating = False

    def run(self, args, ctx: CallContext):
        time.sleep(0.12)
        return "exit code: 3\nstdout:\n"


class AgentStepTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.log = EventLog(f"{self._tmp.name}/l.jsonl")

    def test_each_model_call_is_told_what_the_step_reacts_to(self):
        llm = Recorder(Reply("run_shell", command="npm test"), Reply("finish", answer="done"))
        agent = Agent("main", llm, [SlowShell(), FinishTool()], 5, self.log, Console())
        result = agent.run([{"role": "system", "content": "S"}, GOAL])
        self.assertEqual(result.status, FINISHED)
        self.assertEqual(llm.doing, ['starting on "make me a bmw part picker page"',
                                     "reacting to `npm test` (exit 3)"])

    def test_a_running_command_gets_a_live_timer(self):
        console = mock.Mock(spec=Console)
        console.status.return_value = True
        llm = Recorder(Reply("run_shell", command="npm test"), Reply("finish", answer="done"))
        with mock.patch("agent.agent.working", functools.partial(working, tick=0.02)):
            Agent("main", llm, [SlowShell(), FinishTool()], 5, self.log, console).run([GOAL])
        shown = [c[0][0] for c in console.status.call_args_list]
        self.assertTrue(any(s.startswith("main, step 1: running `npm test`...") for s in shown), shown)

    def test_tools_that_start_nested_agents_get_no_timer_of_their_own(self):
        class Nested(SlowShell):
            name, delegating = "delegate", True
        llm = Recorder(Reply("delegate"), Reply("finish", answer="done"))
        spy = mock.Mock(side_effect=lambda *a, **k: contextlib.nullcontext())
        with mock.patch("agent.agent.working", spy):
            Agent("main", llm, [Nested(), FinishTool()], 5, self.log, Console()).run([GOAL])
        spy.assert_called_once()          # only the finish call; the delegating tool was skipped
        self.assertIn("running", spy.call_args[0][1])
        self.assertNotIn("delegate", spy.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
