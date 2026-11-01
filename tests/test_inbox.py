"""Queued messages: the tiny line editor, the commands, the chat loop, and a real terminal."""
import contextlib
import io
import json
import os
import select
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.app import App                   # noqa: E402
from agent.config import Config             # noqa: E402
from agent.console import Console           # noqa: E402
from agent.inbox import Inbox               # noqa: E402

ROOT = str(Path(__file__).resolve().parent.parent)

try:
    import fcntl
    import pty
    import termios
    HAVE_PTY = True
except ImportError:                         # pragma: no cover
    HAVE_PTY = False


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class EditorTest(unittest.TestCase):
    def setUp(self):
        self.console = mock.Mock()
        self.box = Inbox(self.console)

    def type(self, text):
        self.box.feed(text)

    def notes(self):
        return [c[0][0] for c in self.console.note.call_args_list]

    def test_enter_queues_a_message_and_says_so(self):
        self.type("fix the typo\r")
        self.assertEqual(self.box.items(), ["fix the typo"])
        self.assertEqual(self.notes(), ["[queued 1] fix the typo"])
        self.type("then add tests\n")
        self.assertEqual(self.box.items(), ["fix the typo", "then add tests"])
        self.assertEqual(self.notes()[-1], "[queued 2] then add tests")

    def test_messages_come_out_in_the_order_they_went_in(self):
        self.type("a\rb\rc\r")
        self.assertEqual([self.box.pop(), self.box.pop(), self.box.pop(), self.box.pop()], ["a", "b", "c", None])

    def test_blank_lines_are_ignored(self):
        self.type("\r   \r\r")
        self.assertEqual(self.box.items(), [])
        self.console.note.assert_not_called()

    def test_editing_keys(self):
        self.type("hellp\x7fo")                      # a typo, backspace, retype
        self.assertEqual(self.box._buf, "hello")
        self.type("\x15")                           # Ctrl-U clears the line
        self.assertEqual(self.box._buf, "")
        self.type("one two three\x17")              # Ctrl-W deletes the last word
        self.assertEqual(self.box._buf, "one two ")
        self.type("\x17\x17")
        self.assertEqual(self.box._buf, "")
        self.type("\x7f\x7f")                       # backspace on an empty line is harmless
        self.assertEqual(self.box._buf, "")

    def test_arrow_keys_and_other_escape_sequences_are_swallowed(self):
        self.type("ab\x1b[Ac\x1b[1;5Dd\x1bOPe\x1b[3~f")
        self.assertEqual(self.box._buf, "abcdef")
        self.type("\x1bb")                          # Alt-b: the b belongs to the key, not the text
        self.assertEqual(self.box._buf, "abcdef")

    def test_control_characters_never_end_up_in_the_text(self):
        self.type("a\x00\x01\x04\x07b")
        self.assertEqual(self.box._buf, "ab")

    def test_unicode_is_kept(self):
        self.type("héllo wörld ✓\r")
        self.assertEqual(self.box.items(), ["héllo wörld ✓"])

    def test_a_very_long_line_is_capped(self):
        self.type("x" * 2000)
        self.assertEqual(len(self.box._buf), Inbox.MAX_TYPED)

    def test_queue_command_lists_and_drop_clears(self):
        self.type("/queue\r")
        self.assertEqual(self.notes()[-1], "[queue is empty]")
        self.type("first\rsecond\r/queue\r")
        listing = self.notes()[-1]
        self.assertIn("1. first", listing)
        self.assertIn("2. second", listing)
        self.type("/drop\r")
        self.assertEqual(self.notes()[-1], "[dropped 2 queued message(s)]")
        self.assertEqual(self.box.items(), [])
        self.assertNotIn("/queue", [i for i in self.box.items()])      # commands are never queued

    def test_a_full_queue_refuses_more(self):
        for i in range(Inbox.MAX):
            self.box.submit(f"m{i}")
        self.box.submit("one too many")
        self.assertEqual(len(self.box), Inbox.MAX)
        self.assertIn("queue full", self.notes()[-1])

    def test_the_status_suffix_shows_the_count_and_the_typed_text(self):
        self.assertEqual(self.box._suffix(), "> _")
        self.type("hel")
        self.assertEqual(self.box._suffix(), "> hel_")
        self.type("lo\rnext")
        self.assertEqual(self.box._suffix(), "1 queued | > next_")

    def test_the_status_line_is_redrawn_as_you_type(self):
        self.type("a")
        self.console.refresh_status.assert_called()

    def test_without_a_terminal_nothing_is_captured(self):
        box = Inbox(self.console, stdin=io.StringIO())
        self.assertFalse(box.supported)
        with box.capturing():
            pass
        self.console.set_status_suffix.assert_not_called()


class ConsoleSuffixTest(unittest.TestCase):
    def run_status(self, columns, suffix, text="main, step 2: reacting to `npm test` (exit 1), waiting for the model... 12s"):
        err = FakeTTY()
        with mock.patch.object(sys, "stderr", err), \
                mock.patch("agent.console.shutil.get_terminal_size", return_value=os.terminal_size((columns, 24))):
            c = Console()
            c.set_status_suffix(lambda: suffix)
            c.status(text)
        return err.getvalue().rsplit("\r\x1b[K", 1)[-1]

    def test_the_suffix_follows_the_status(self):
        drawn = self.run_status(200, "> fix the typo_")
        self.assertTrue(drawn.endswith(" | > fix the typo_"))
        self.assertIn("waiting for the model... 12s", drawn)

    def test_on_a_narrow_terminal_the_typed_text_wins_over_the_status(self):
        drawn = self.run_status(50, "> fix the typo_")
        self.assertLessEqual(len(drawn), 49)
        self.assertTrue(drawn.endswith("> fix the typo_"))
        self.assertIn("...", drawn)

    def test_a_very_long_typed_line_shows_its_end(self):
        drawn = self.run_status(60, "> " + "a" * 80 + "TAIL_")
        self.assertLessEqual(len(drawn), 59)
        self.assertTrue(drawn.endswith("TAIL_"))

    def test_other_output_does_not_make_the_typing_line_vanish(self):
        err, out = FakeTTY(), io.StringIO()
        with mock.patch.object(sys, "stderr", err), mock.patch.object(sys, "stdout", out):
            c = Console()
            c.set_status_suffix(lambda: "> typed_")
            c.status("working")
            c.info("a result")
            self.assertTrue(err.getvalue().endswith("working | > typed_"))   # drawn again after the output
            c.set_status_suffix(None)
            c.clear_status()
            before = len(err.getvalue())
            c.status("working")
            c.info("another")
            self.assertEqual(err.getvalue()[before:].count("working"), 1)      # no suffix: not redrawn


class ChatLoopTest(unittest.TestCase):
    """The queue runs in order, without the prompt in between."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        t = self._tmp.name
        self.app = App(Config.from_env({"WORKSPACE": f"{t}/ws", "SESSION_FILE": f"{t}/s.json",
                                        "LOG_FILE": f"{t}/l.jsonl", "FRIEND_MEMORY_FILE": f"{t}/m.json",
                                        "SANDBOX_MODE": "none", "LLM_URL": "http://x/v1"}), llm=mock.Mock())

    def chat(self, typed, queued_during_first=(), preload=()):
        """Run chat() with these prompt answers; `queued_during_first` is typed while goal 1 runs."""
        ran, first = [], [True]

        class TestInbox(Inbox):
            @contextlib.contextmanager
            def capturing(inbox):
                if first[0]:                        # the user types these while the first goal runs
                    first[0] = False
                    for m in queued_during_first:
                        inbox.submit(m)
                yield inbox

        def run_goal(goal):
            ran.append(goal)

        out = io.StringIO()
        with mock.patch("agent.app.Inbox", TestInbox), mock.patch.object(self.app, "run_goal", run_goal), \
                mock.patch("builtins.input", side_effect=list(typed)) as inp, contextlib.redirect_stdout(out), \
                mock.patch.object(Console, "note", lambda self, text: None), mock.patch.object(Console, "refresh_status", lambda self: None):
            orig_init = TestInbox.__init__

            def init(box, *a, **k):
                orig_init(box, *a, **k)
                for m in preload:
                    box._items.append(m)
            with mock.patch.object(TestInbox, "__init__", init):
                self.app.chat()
        return ran, inp.call_count, out.getvalue()

    def test_queued_messages_run_in_order_after_the_current_goal(self):
        ran, prompts, out = self.chat(["first", "exit"], queued_during_first=["queued A", "queued B"])
        self.assertEqual(ran, ["first", "queued A", "queued B"])
        self.assertEqual(prompts, 2)                        # the prompt only came back when the queue was empty
        self.assertIn("goal (queued)> queued A", out)
        self.assertIn("goal (queued)> queued B", out)

    def test_exit_in_the_queue_ends_the_session_after_what_came_before(self):
        ran, prompts, _ = self.chat(["first"], queued_during_first=["more work", "exit", "never runs"])
        self.assertEqual(ran, ["first", "more work"])
        self.assertEqual(prompts, 1)

    def test_commands_work_at_the_prompt(self):
        ran, _, out = self.chat(["/queue", "/drop", "/queue", "exit"], preload=["waiting one"])
        # preloaded items run first (they are the queue); commands then see an empty queue
        self.assertEqual(ran, ["waiting one"])
        self.assertIn("[queue is empty]", out)

    def test_plain_chat_still_works(self):
        ran, prompts, _ = self.chat(["a", "", "b", "quit"])
        self.assertEqual(ran, ["a", "b"])
        self.assertEqual(prompts, 4)


def _winsize(fd, rows=24, cols=100):
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))


def _drive(script: str, keystrokes: list, ready_marker="READY", timeout=15, head_start=0.0):
    """Run `script` on a real pseudo-terminal, send `keystrokes` once it prints the marker, and
    return everything it wrote."""
    master, slave = pty.openpty()
    _winsize(slave)
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=slave, stdout=slave, stderr=slave,
                            close_fds=True, cwd=ROOT)
    os.close(slave)
    out, sent, deadline = b"", False, time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.05)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                out += chunk
            if not sent and ready_marker.encode() in out:
                sent = True
                time.sleep(head_start)              # let the program draw a few frames first
                for k in keystrokes:
                    os.write(master, k.encode())
                    time.sleep(0.05)
            if proc.poll() is not None and not ready:
                break
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
    return out.decode(errors="replace")


@unittest.skipUnless(HAVE_PTY, "needs a pseudo-terminal")
class RealTerminalTest(unittest.TestCase):
    def test_keystrokes_during_a_run_are_queued_and_the_terminal_is_restored(self):
        script = textwrap.dedent("""
            import json, sys, termios, time
            from agent.console import Console
            from agent.inbox import Inbox
            c = Console(); box = Inbox(c); fd = sys.stdin.fileno()
            with box.capturing():
                f = termios.tcgetattr(fd)[3]
                inside = [bool(f & termios.ECHO), bool(f & termios.ICANON), bool(f & termios.ISIG)]
                print("READY", flush=True)
                time.sleep(1.5)
            f = termios.tcgetattr(fd)[3]
            print(json.dumps({"items": box.items(), "leftover": box.leftover, "inside": inside,
                              "echo_after": bool(f & termios.ECHO), "canon_after": bool(f & termios.ICANON)}))
        """)
        out = _drive(script, ["first message\r", "second one\r", "half typ", "ed"])
        line = next(l for l in out.replace("\r", "\n").splitlines() if l.startswith('{"items"'))
        got = json.loads(line)
        self.assertEqual(got["items"], ["first message", "second one"])
        self.assertEqual(got["leftover"], "half typed")                   # offered at the next prompt
        self.assertEqual(got["inside"], [False, False, True])             # no echo, no line mode, Ctrl-C still works
        self.assertTrue(got["echo_after"] and got["canon_after"])         # restored for input()
        self.assertIn("[queued 1] first message", out)
        self.assertIn("[queued 2] second one", out)
        self.assertNotIn("first message\r\nfirst message", out)           # not echoed twice

    def test_a_half_typed_line_survives_the_next_queued_run(self):
        script = textwrap.dedent("""
            import json, sys, time
            from agent.console import Console
            from agent.inbox import Inbox
            c = Console(); box = Inbox(c)
            with box.capturing():                   # goal 1
                print("READY", flush=True)
                time.sleep(1.0)
            first = box.leftover
            with box.capturing():                   # goal 2, queued behind it: nothing typed now
                time.sleep(0.3)
            print(json.dumps({"after_first": first, "after_second": box.leftover}))
        """)
        out = _drive(script, ["half typed"])
        line = next(l for l in out.replace("\r", "\n").splitlines() if l.startswith('{"after_first"'))
        self.assertEqual(json.loads(line), {"after_first": "half typed", "after_second": "half typed"})

    def test_what_you_type_appears_in_the_status_line(self):
        script = textwrap.dedent("""
            import time
            from agent.console import Console, working
            from agent.inbox import Inbox
            c = Console(); box = Inbox(c)
            c.start_clock()
            with box.capturing():
                with working(c, "main, step 1: waiting for the model...", tick=0.05):
                    print("READY", flush=True)
                    time.sleep(1.2)
            print("DONE")
        """)
        out = _drive(script, ["fix the typo"], head_start=0.4)
        frames = [f for f in out.split("\r\x1b[K") if "waiting for the model" in f]
        self.assertTrue(any("> fix the typo_" in f for f in frames), frames[-3:])
        self.assertTrue(any(f.rstrip().endswith("> _") for f in frames), "an empty prompt before typing")
        self.assertIn("DONE", out)

    def test_arrow_keys_do_not_corrupt_the_line(self):
        script = textwrap.dedent("""
            import json, time
            from agent.console import Console
            from agent.inbox import Inbox
            c = Console(); box = Inbox(c)
            with box.capturing():
                print("READY", flush=True)
                time.sleep(1.0)
            print(json.dumps(box.items()))
        """)
        out = _drive(script, ["ab\x1b[A\x1b[B\x1b[C\x1b[Dc\r"])
        line = next(l for l in out.replace("\r", "\n").splitlines() if l.startswith("[") and l.endswith("]") and "abc" in l)
        self.assertEqual(json.loads(line), ["abc"])


if __name__ == "__main__":
    unittest.main()
