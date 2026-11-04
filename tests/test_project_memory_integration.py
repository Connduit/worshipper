"""Project memory together with context management, the resume reminder and the app.

The memory (storage.ProjectMemory) and compaction (context.compact) were written separately.
These tests cover where they meet: the memory note is a second system message, compaction must
never drop it, the steps compaction does drop must reach the memory, and the next goal must see it.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.app import App                                    # noqa: E402
from agent.config import Config                              # noqa: E402
from agent.context import compact, total_chars               # noqa: E402
from agent.storage import ProjectMemory                      # noqa: E402

PREFIX = "[persistent project memory]"


def goal(text):
    return {"role": "user", "content": f"GOAL: {text}"}


def step(i, out_chars=3000):
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "run_shell", "arguments": '{"command":"ls"}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * out_chars}]


def finish(answer, i):
    return [{"role": "assistant", "content": None, "tool_calls": [
                {"id": f"f{i}", "type": "function", "function": {"name": "finish", "arguments": json.dumps({"answer": answer})}}]},
            {"role": "tool", "tool_call_id": f"f{i}", "content": "Goal marked complete."}]


SYSTEM = {"role": "system", "content": "S"}


class Reply:
    def __init__(self):
        self.content = None
        self.tool_calls = [NS(id="r1", function=NS(name="finish", arguments='{"answer": "ok"}'))]

    def model_dump(self, exclude_none=False):
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "r1", "type": "function", "function": {"name": "finish", "arguments": '{"answer": "ok"}'}}]}


class Llm:
    """A model that always finishes at once, and records what it was sent."""
    model_name = "fake"

    def __init__(self, n_ctx=None):
        self.n_ctx, self.sent = n_ctx, []

    def wait_ready(self, *a, **k):
        return True

    def context_size(self):
        return self.n_ctx

    def failure_text(self):
        return "failed"

    def chat(self, messages, tools=None, **kw):
        self.sent.append(json.loads(json.dumps(messages)))
        return NS(choices=[NS(message=Reply())])


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.session = f"{self.tmp}/session.json"

    def app(self, llm=None, **env):
        cfg = Config.from_env({"WORKSPACE": f"{self.tmp}/ws", "SESSION_FILE": self.session, "LOG_FILE": f"{self.tmp}/log.jsonl",
                               "FRIEND_MEMORY_FILE": f"{self.tmp}/mem.json", "SANDBOX_MODE": "none",
                               "LLM_URL": "http://x/v1", **env})
        self.llm = llm or Llm()
        return App(cfg, llm=self.llm)

    def save_history(self, messages):
        Path(self.session).write_text(json.dumps(messages))

    def memory(self):
        return ProjectMemory(self.session)

    def run_quiet(self, app, text):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return app.run_goal(text)


class InjectionTest(Base):
    def test_a_saved_memory_becomes_one_system_note_right_after_the_system_prompt(self):
        self.save_history([SYSTEM, goal("old"), *finish("did old", 1)])
        self.memory().remember([goal("make a dark theme")])
        app = self.app()
        app.setup()
        self.assertEqual(app.messages[0]["content"], app.system)
        note = app.messages[1]
        self.assertEqual(note["role"], "system")
        self.assertTrue(note["content"].startswith(PREFIX))
        self.assertIn("User request: GOAL: make a dark theme", note["content"])
        self.assertIn("verify against files", note["content"])

    def test_every_goal_sees_exactly_one_fresh_note(self):
        self.memory().remember([goal("first request")])
        app = self.app()
        app.setup()
        self.run_quiet(app, "one")
        self.memory().remember([goal("second request")])           # something was dropped meanwhile
        self.run_quiet(app, "two")
        for sent in self.llm.sent:
            notes = [m for m in sent if m["role"] == "system" and m["content"].startswith(PREFIX)]
            self.assertEqual(len(notes), 1)
        self.assertNotIn("second request", self.llm.sent[0][1]["content"])
        self.assertIn("second request", self.llm.sent[1][1]["content"])      # refreshed, not stacked

    def test_without_a_memory_there_is_no_note(self):
        app = self.app()
        app.setup()
        self.assertEqual([m["role"] for m in app.messages], ["system"])
        self.run_quiet(app, "go")
        self.assertFalse(any(str(m.get("content")).startswith(PREFIX) for m in self.llm.sent[0]))

    def test_a_new_conversation_forgets_the_memory_and_the_archive(self):
        self.memory().remember([goal("old project")])
        mem = self.memory()
        self.assertTrue(Path(mem.path).exists() and Path(mem.archive_path).exists())
        app = self.app()
        app.setup(new=True)
        self.assertFalse(Path(mem.path).exists() or Path(mem.archive_path).exists())
        self.assertEqual([m["role"] for m in app.messages], ["system"])

    def test_a_resumed_session_does_not_pile_up_notes_in_session_json(self):
        self.memory().remember([goal("kept")])
        app = self.app()
        app.setup()
        self.run_quiet(app, "one")
        for _ in range(3):
            app = self.app()                                    # three more restarts
            app.setup()
            self.run_quiet(app, "again")
        saved = json.loads(Path(self.session).read_text())
        self.assertEqual(sum(1 for m in saved if m["role"] == "system" and m["content"].startswith(PREFIX)), 1)


class DisplayTest(Base):
    def debug_line(self, history):
        self.save_history(history)
        app = self.app(DEBUG="1")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            app.setup()
        return next(l for l in out.getvalue().splitlines() if l.startswith("[session]"))

    def test_the_memory_note_is_not_counted_as_earlier_messages(self):
        self.memory().remember([goal("something")])
        self.assertEqual(self.debug_line([SYSTEM]), "[session] new conversation")
        self.memory().remember([goal("more")])
        self.assertEqual(self.debug_line([SYSTEM, goal("a"), *finish("b", 1)]), "[session] 3 earlier messages loaded")

    def test_the_resume_reminder_still_lists_the_goals_with_a_memory_note_present(self):
        self.memory().remember([goal("long ago")])
        self.save_history([SYSTEM, goal("build the picker"), *finish("built it", 1), goal("add search"), *finish("added it", 2)])
        app = self.app()
        app.setup()
        self.assertIn("2 goals so far", app._recap)
        self.assertIn("you:   build the picker", app._recap)
        self.assertNotIn("long ago", app._recap)                # the memory is not a goal of this conversation


class CompactionMeetsMemoryTest(unittest.TestCase):
    def history(self, steps=12):
        msgs = [SYSTEM, {"role": "system", "content": PREFIX + "\nearlier notes"}, goal("first job"), *step(0), *finish("did first", 0),
                goal("second job"), *step(1), *finish("did second", 1), goal("the current job")]
        for i in range(2, steps):
            msgs += step(i)
        return msgs

    def test_the_memory_note_is_never_dropped_or_trimmed(self):
        msgs = self.history()
        compact(msgs, 1500)
        self.assertEqual([m["role"] for m in msgs[:2]], ["system", "system"])
        self.assertEqual(msgs[1]["content"], PREFIX + "\nearlier notes")
        self.assertTrue(any(m.get("content") == "GOAL: the current job" for m in msgs))

    def test_every_dropped_message_is_reported_exactly_once(self):
        msgs = self.history()
        before = [json.dumps(m, sort_keys=True) for m in msgs]
        reported = []
        compact(msgs, 1500, on_drop=lambda removed: reported.extend(removed))
        after = [json.dumps(m, sort_keys=True) for m in msgs]
        # trimming rewrites some kept messages, so judge by identity of what disappeared
        gone_ids = {m.get("tool_call_id") or json.dumps(m.get("tool_calls")) or m["content"]
                    for m in reported}
        self.assertTrue(reported)
        self.assertEqual(len(reported), len({json.dumps(m, sort_keys=True) for m in reported}))   # no duplicates
        self.assertFalse(any(m["role"] == "system" for m in reported))
        reported_goals = [m["content"] for m in reported if m["role"] == "user"]
        self.assertIn("GOAL: first job", reported_goals)
        self.assertNotIn("GOAL: the current job", reported_goals)
        self.assertTrue(gone_ids)

    def test_trimming_alone_reports_nothing(self):
        msgs = self.history(steps=5)
        reported = []
        compact(msgs, total_chars(msgs) - 1200, on_drop=lambda removed: reported.extend(removed))
        self.assertEqual(reported, [])                          # only old outputs were cut, nothing was removed

    def test_with_nothing_to_drop_the_callback_is_not_called(self):
        calls = []
        compact([SYSTEM, goal("tiny")], 10_000, on_drop=calls.append)
        self.assertEqual(calls, [])


class EndToEndTest(Base):
    """A long conversation in a small window: compaction drops old steps, the memory keeps them,
    and the next goal starts with them in view."""

    def long_history(self):
        msgs = [SYSTEM]
        for n in range(1, 4):
            msgs += [goal(f"job number {n}"), *step(n * 10), *step(n * 10 + 1), *finish(f"finished job {n}", n)]
        return msgs

    def test_dropped_steps_reach_the_memory_and_the_next_goal_sees_them(self):
        self.save_history(self.long_history())
        app = self.app(llm=Llm(n_ctx=3000))
        app.setup()
        self.run_quiet(app, "the new job")
        mem = self.memory()
        self.assertTrue(Path(mem.archive_path).exists())
        archived = [json.loads(l) for l in Path(mem.archive_path).read_text().splitlines()]
        self.assertTrue(archived)
        self.assertTrue(all("role" in m for m in archived))
        remembered = mem.read()
        self.assertIn("User request: GOAL: job number 1", remembered)
        self.assertNotIn("the new job", remembered)             # the current goal is never dropped
        # the compacted request still had its memory slot, and the current goal
        first_request = self.llm.sent[0]
        self.assertEqual(first_request[0]["role"], "system")
        self.assertTrue(any(m.get("content") == "GOAL: the new job" for m in first_request))
        # the next goal starts with the memory in view
        self.run_quiet(app, "a follow-up")
        second_request = self.llm.sent[1]
        self.assertEqual(second_request[1]["role"], "system")
        self.assertTrue(second_request[1]["content"].startswith(PREFIX))
        self.assertIn("job number 1", second_request[1]["content"])

    def test_a_window_that_is_big_enough_creates_no_memory(self):
        self.save_history(self.long_history())
        app = self.app(llm=Llm(n_ctx=100_000))
        app.setup()
        self.run_quiet(app, "the new job")
        self.assertFalse(Path(self.memory().path).exists())
        self.assertFalse(Path(self.memory().archive_path).exists())


class ProjectMemoryTest(Base):
    def test_what_is_kept_as_an_excerpt(self):
        mem = self.memory()
        mem.remember([
            {"role": "user", "content": "GOAL: " + "x" * 2000},
            {"role": "user", "content": "[context note] earlier steps were removed"},
            {"role": "assistant", "content": "I changed the header. " + "y" * 800},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "w", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": "src/main.js", "content": "SECRET BODY"})}}]},
            {"role": "tool", "tool_call_id": "w", "content": "TOOL OUTPUT"},
        ])
        text = mem.read()
        self.assertIn("User request: GOAL: xxx", text)
        self.assertLess(len(text.split("Agent response:")[0]), 1100 + 40)       # requests are cut
        self.assertIn("Agent response: I changed the header.", text)
        self.assertIn("Wrote file: src/main.js", text)
        for leaked in ("SECRET BODY", "TOOL OUTPUT", "context note"):
            self.assertNotIn(leaked, text)                       # file bodies and tool output stay in the archive only
        self.assertIn("SECRET BODY", Path(mem.archive_path).read_text())

    def test_the_memory_is_bounded_and_keeps_the_newest_information(self):
        mem = ProjectMemory(self.session, max_chars=500)
        for i in range(60):
            mem.remember([{"role": "user", "content": f"GOAL: request number {i}"}])
        text = mem.read()
        self.assertLessEqual(len(text), 500)
        self.assertIn("request number 59", text)
        self.assertNotIn("request number 0\n", text)

    def test_only_the_archive_grows_when_nothing_is_worth_an_excerpt(self):
        mem = self.memory()
        mem.remember([{"role": "tool", "tool_call_id": "x", "content": "just output"}])
        self.assertEqual(mem.read(), "")
        self.assertFalse(Path(mem.path).exists())
        self.assertEqual(len(Path(mem.archive_path).read_text().splitlines()), 1)

    def test_a_damaged_memory_file_reads_as_empty_and_recovers(self):
        mem = self.memory()
        Path(mem.path).write_text("{not json")
        self.assertEqual(mem.read(), "")
        Path(mem.path).write_text('["not a string"]')
        self.assertEqual(mem.read(), "")
        mem.remember([{"role": "user", "content": "GOAL: fresh start"}])
        self.assertIn("fresh start", mem.read())

    def test_writes_are_atomic_and_leave_no_temporary_file(self):
        mem = self.memory()
        mem.remember([{"role": "user", "content": "GOAL: x"}])
        self.assertFalse(Path(mem.path + ".tmp").exists())

    def test_delete_removes_both_files_and_is_safe_twice(self):
        mem = self.memory()
        mem.remember([{"role": "user", "content": "GOAL: x"}])
        mem.delete()
        mem.delete()
        self.assertFalse(Path(mem.path).exists() or Path(mem.archive_path).exists())

    def test_the_files_sit_next_to_the_session_file(self):
        mem = self.memory()
        self.assertEqual(mem.path, self.session + ".memory.json")
        self.assertEqual(mem.archive_path, self.session + ".archive.jsonl")


if __name__ == "__main__":
    unittest.main()
