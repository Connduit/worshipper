"""Unit tests. No llama-server, bwrap or `openai` package needed.

    python -m unittest discover -s tests -q        (run from the code root)
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import cli                                          # noqa: E402
from agent.agent import FINISHED, LLM_FAILED, MAX_STEPS, Agent  # noqa: E402
from agent.app import App                                      # noqa: E402
from agent.config import Config                                # noqa: E402
from agent.devcopy import DevCopy                              # noqa: E402
from agent.friends import FriendRegistry, FriendSpec           # noqa: E402
from agent.llm import LLM                                      # noqa: E402
from agent.sandbox import BwrapSandbox, LlamaBridge, NoSandbox, make_sandbox  # noqa: E402
from agent.storage import EventLog, FriendMemory               # noqa: E402


# --------------------------------------------------------------------------- fakes
class Msg:
    """Looks like an OpenAI chat message."""

    def __init__(self, content=None, calls=()):
        self.content = content
        self.tool_calls = [NS(id=f"c{i}", function=NS(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
                           for i, (n, a) in enumerate(calls)] or None

    def model_dump(self, exclude_none=True):
        d = {"role": "assistant"}
        if self.content:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


class FakeLLM:
    """Replays scripted messages in order and records every call. Stands in for LLM."""
    base_url, model = "http://fake/v1", "fake"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None, *, depth=0, label="main", step=None):
        self.calls.append({"label": label, "depth": depth, "messages": [dict(m) for m in messages],
                           "tools": [t["function"]["name"] for t in tools] if tools else None})
        item = self.script.pop(0)
        return None if item is None else NS(choices=[NS(message=item)])


def call(name, **args):
    return (name, args)


def env_for(tmp, **extra):
    return {"WORKSPACE": f"{tmp}/ws", "SESSION_FILE": f"{tmp}/s.json", "LOG_FILE": f"{tmp}/l.jsonl",
            "FRIENDS_FILE": f"{tmp}/friends.json", "SANDBOX_MODE": "none", "MAX_STEPS": "10", **extra}


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        out = contextlib.redirect_stdout(io.StringIO())     # keep test output quiet
        out.__enter__()
        self.addCleanup(out.__exit__, None, None, None)

    def make_app(self, llm, **extra):
        app = App(Config.from_env(env_for(self.tmp, **extra)), llm=llm)
        app.setup(new=True)
        return app


# --------------------------------------------------------------------------- tests
class AgentFlowTest(TmpCase):
    def test_advisors_workers_and_errors(self):
        target = f"{self.tmp}/f.txt"
        llm = FakeLLM(
            Msg(None, [call("ask_friend", friend="reviewer", message="review plan X")]),   # main
            Msg("Looks fine, check edge cases."),                                         # advisor
            Msg(None, [call("ask_friend", friend="coder", message="hi")]),                # wrong kind
            Msg(None, [call("delegate", friend="coder", task="write f.txt")]),            # main
            Msg(None, [call("run_shell", command=f"echo hi > {target}")]),                # worker
            Msg(None, [call("delegate", friend="coder", task="nested")]),                 # not offered
            Msg(None, [call("finish", summary="wrote f.txt")]),                           # worker
            Msg(None, [call("delegate", friend="nobody", task="x")]),                     # unknown
            Msg("no tool call here"),                                                     # nudged
            Msg(None, [call("finish", summary="all done")]),
        )
        app = self.make_app(llm, USE_FRIENDS="1")
        result = app.run_goal("do the thing")

        self.assertEqual((result.status, result.text), (FINISHED, "all done"))
        self.assertEqual(Path(target).read_text().strip(), "hi")
        self.assertEqual(llm.script, [])

        by_label = {}
        for c in llm.calls:
            by_label.setdefault(c["label"], c)
        self.assertEqual(by_label["main"]["tools"], ["get_time", "run_shell", "ask_friend", "delegate", "finish"])
        self.assertIsNone(by_label["reviewer"]["tools"])                       # advisors have no tools
        self.assertEqual(by_label["coder"]["tools"], ["run_shell", "get_time", "finish"])
        self.assertEqual(by_label["coder"]["depth"], 1)

        observations = [m["content"] for m in app.messages if m["role"] == "tool"]
        self.assertIn("Looks fine", observations[0])
        self.assertIn("unknown advisor 'coder'", observations[1])
        self.assertIn("[coder finished]\nwrote f.txt", observations[2])
        self.assertIn("unknown worker 'nobody'", observations[3])
        coder_msgs = [c for c in llm.calls if c["label"] == "coder"][-1]["messages"]
        self.assertTrue(any("not available to you" in str(m.get("content")) for m in coder_msgs))
        self.assertTrue(os.path.exists(app.cfg.session_file))
        self.assertIn("reviewer", json.loads(Path(app.cfg.friend_memory_file).read_text()))

    def test_max_steps_and_llm_failure(self):
        app = self.make_app(FakeLLM(Msg("hmm"), Msg("hmm")), MAX_STEPS="2")
        self.assertEqual(app.run_goal("x").status, MAX_STEPS)
        app = self.make_app(FakeLLM(None))
        self.assertEqual(app.run_goal("x").status, LLM_FAILED)

    def test_bad_arguments(self):
        llm = FakeLLM(Msg(None, [call("run_shell")]), Msg(None, [("run_shell", "{not json")]),
                      Msg(None, [call("finish", summary="ok")]))
        app = self.make_app(llm)
        app.run_goal("x")
        obs = [m["content"] for m in app.messages if m["role"] == "tool"]
        self.assertIn("missing argument 'command'", obs[0])
        self.assertIn("not valid JSON", obs[1])

    def test_shell_timeout(self):
        llm = FakeLLM(Msg(None, [call("run_shell", command="sleep 5", timeout=1)]),
                      Msg(None, [call("finish", summary="ok")]))
        app = self.make_app(llm)
        app.run_goal("x")
        self.assertIn("timed out after 1s", [m["content"] for m in app.messages if m["role"] == "tool"][0])


class FriendConfigTest(TmpCase):
    def test_defaults(self):
        reg = FriendRegistry.load(f"{self.tmp}/missing.json")
        self.assertEqual([f.name for f in reg.advisors()], ["planner", "reviewer", "tester"])
        self.assertEqual([f.name for f in reg.workers()], ["coder", "inspector"])
        self.assertTrue(reg.get("inspector", "worker").readonly)
        self.assertIsNone(reg.get("inspector", "advisor"))

    def test_file_overrides_removes_and_validates(self):
        path = f"{self.tmp}/friends.json"
        Path(path).write_text(json.dumps({
            "tester": None, "planner": {"kind": "advisor", "prompt": "new", "desc": "d"},
            "translator": {"kind": "advisor", "prompt": "x", "url": "http://h:1/v1"},
            "bad": {"kind": "zzz"}, "worse": {"kind": "worker", "prompt": "p", "max_steps": "lots"}}))
        reg = FriendRegistry.load(path)
        self.assertEqual(reg.get("planner", "advisor").prompt, "new")
        self.assertIsNone(reg.get("tester", "advisor"))
        self.assertEqual(reg.get("translator", "advisor").url, "http://h:1/v1")
        self.assertNotIn("bad", reg.names())
        self.assertNotIn("worse", reg.names())

    def test_unreadable_file_falls_back_to_defaults(self):
        path = f"{self.tmp}/friends.json"
        Path(path).write_text("{broken")
        self.assertIn("planner", FriendRegistry.load(path).names())

    def test_solo_by_default_and_opt_in(self):
        def offered(app):
            return [t.name for t in app.pool.select(["ask_friend", "delegate", "run_shell"], 0)]

        app = self.make_app(FakeLLM())                                  # default: solo
        self.assertEqual(offered(app), ["run_shell", "finish"])
        self.assertNotIn("friends", app.system)
        self.assertFalse(app.registry)

        app = self.make_app(FakeLLM(), USE_FRIENDS="1")
        self.assertEqual(offered(app), ["ask_friend", "delegate", "run_shell", "finish"])
        self.assertIn("friends", app.system)

        app = self.make_app(FakeLLM(), USE_FRIENDS="1", MAX_DEPTH="0")
        self.assertEqual(offered(app), ["ask_friend", "run_shell", "finish"])
        self.assertNotIn("Workers", app.system)

    def test_resume_refreshes_system_prompt(self):
        app = self.make_app(FakeLLM(Msg(None, [call("finish", summary="a")])))
        app.run_goal("x")
        solo_prompt = app.system
        app2 = App(Config.from_env(env_for(self.tmp, USE_FRIENDS="1")), llm=FakeLLM())
        app2.setup()                                                    # resumes the saved session
        self.assertGreater(len(app2.messages), 1)
        self.assertEqual(app2.messages[0]["content"], app2.system)
        self.assertNotEqual(app2.messages[0]["content"], solo_prompt)

    def test_friend_with_own_url_gets_own_llm(self):
        made = []
        app = self.make_app(FakeLLM(Msg("hello")))
        app.service.registry = FriendRegistry({"t": FriendSpec("t", "advisor", "p", url="http://other/v1", model="m2")})
        app.service._make_llm = lambda url, model: made.append((url, model)) or FakeLLM(Msg("from t"))
        self.assertEqual(app.service.ask("t", "hi", 0), "from t")
        self.assertEqual(made, [("http://other/v1", "m2")])


class FriendMemoryTest(TmpCase):
    def test_trim_persist_and_rollback(self):
        mem = FriendMemory(f"{self.tmp}/m.json", max_history=12)
        for n in range(10):
            mem.prepare("a", "sys", f"m{n}")
            mem.commit("a", "ok")
        h = mem.history("a")
        self.assertEqual((len(h), h[0]["role"], h[1]["role"], h[1]["content"]), (13, "system", "user", "m4"))
        mem.prepare("a", "NEW SYSTEM", "unanswered")
        mem.rollback("a")
        self.assertEqual(mem.history("a")[0]["content"], "NEW SYSTEM")   # always the current prompt
        self.assertEqual(mem.history("a")[-1]["role"], "assistant")
        again = FriendMemory(f"{self.tmp}/m.json", 12)
        again.load()
        self.assertEqual(len(again.history("a")), 13)

    def test_ignores_garbage_file(self):
        Path(f"{self.tmp}/m.json").write_text(json.dumps({"a": "nope", "b": [1], "c": [{"role": "user"}]}))
        mem = FriendMemory(f"{self.tmp}/m.json", 12)
        mem.load()
        self.assertEqual((mem.history("a"), mem.history("b")), ([], []))
        self.assertEqual(len(mem.history("c")), 1)


class SandboxTest(TmpCase):
    def test_bwrap_readonly_and_network(self):
        sb = BwrapSandbox("/some/ws")
        rw, ro = sb.command("ls"), sb.command("ls", readonly=True)
        self.assertEqual(rw[rw.index("/some/ws") - 1], "--bind")
        self.assertEqual(ro[ro.index("/some/ws") - 1], "--ro-bind")
        self.assertIn("--unshare-all", rw)
        self.assertNotIn("--share-net", rw)

    def test_bridge_only_exposes_socket_dir(self):
        bridge = LlamaBridge(8080)
        bridge.sock_dir = "/tmp/agent-sock-x"
        argv = BwrapSandbox("/ws", bridge).command("ls")
        self.assertEqual(argv[argv.index("/run/llama") - 2:argv.index("/run/llama") + 1],
                         ["--ro-bind", "/tmp/agent-sock-x", "/run/llama"])
        self.assertIn("TCP-LISTEN:8080", argv[-1])

    def test_make_sandbox(self):
        self.assertIsInstance(make_sandbox(Config.from_env(env_for(self.tmp))), NoSandbox)
        self.assertIsInstance(make_sandbox(Config.from_env(env_for(self.tmp, SANDBOX_MODE="bwrap"))), BwrapSandbox)
        self.assertIsInstance(make_sandbox(Config.from_env(env_for(self.tmp, SANDBOX_MODE="typo"))), BwrapSandbox)

    def test_none_sandbox_runs(self):
        self.assertEqual(NoSandbox().run("echo hi", 5).stdout.strip(), "hi")


class LLMRetryTest(TmpCase):
    class ConnErr(Exception):
        pass

    def make_llm(self, create):
        stub = types.ModuleType("openai")
        stub.APIStatusError, stub.APIConnectionError = RuntimeError, self.ConnErr
        stub.OpenAI = lambda **kw: NS(chat=NS(completions=NS(create=create)))
        with mock.patch.dict(sys.modules, {"openai": stub}):
            return LLM("http://x/v1", "m", EventLog(f"{self.tmp}/l.jsonl"))

    def test_retries_with_rising_temperature(self):
        temps = []

        def create(**kw):
            temps.append(kw["temperature"])
            if len(temps) < 3:
                raise self.ConnErr("boom")
            return "ok"

        self.assertEqual(self.make_llm(create).chat([{"role": "user", "content": "hi"}]), "ok")
        self.assertEqual(temps, [0.2, 0.7, 1.0])

    def test_gives_up_after_three_failures(self):
        create = mock.Mock(side_effect=self.ConnErr("down"))
        self.assertIsNone(self.make_llm(create).chat([]))
        self.assertEqual(create.call_count, 3)
        self.assertEqual(Path(f"{self.tmp}/l.jsonl").read_text().count("llm_error"), 3)


class DevCopyTest(TmpCase):
    def test_sync_diff_and_reset(self):
        src, dest = Path(self.tmp, "src"), Path(self.tmp, "dest")
        (src / "agent").mkdir(parents=True)
        (src / "agent_local.py").write_text("a = 1\n")
        (src / "agent" / "m.py").write_text("x = 1\n")
        dc = DevCopy(src, dest)
        with self.assertRaises(FileNotFoundError):
            dc.diff()
        dc.sync()
        self.assertEqual(dc.diff(), [])
        (dest / "agent" / "m.py").write_text("x = 2\n")
        (dest / "agent" / "new.py").write_text("y = 1\n")
        diff = "".join(dc.diff())
        self.assertIn("-x = 1", diff)
        self.assertIn("+x = 2", diff)
        self.assertIn("agent/new.py (model)", diff)
        dc.sync()                                   # no force: keeps the model's edits
        self.assertEqual((dest / "agent" / "m.py").read_text(), "x = 2\n")
        dc.sync(force=True)
        self.assertEqual(dc.diff(), [])
        self.assertFalse((dest / "agent" / "new.py").exists())


class CliTest(TmpCase):
    def run_cli(self, *argv, env=None):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, env or {}), contextlib.redirect_stdout(buf):
            cli.main(list(argv))
        return buf.getvalue()

    def test_help_only_describes_flags(self):
        out = self.run_cli("-h")
        for flag in ("--new", "--chat", "--friends", "--list-friends", "--sync", "--diff"):
            self.assertIn(flag, out)
        self.assertIn("README.md", out)
        for noise in ("MAX_DEPTH", "SANDBOX_MODE", "pacman", "friends.json"):
            self.assertNotIn(noise, out)
        self.assertLess(len(out.splitlines()), 35)
        self.assertEqual(self.run_cli("--help"), out)

    def test_list_friends_works_even_though_friends_are_off(self):
        out = self.run_cli("--list-friends", env={**env_for(self.tmp), "USE_FRIENDS": "0"})
        self.assertIn("inspector", out)
        self.assertIn("readonly", out)
        self.assertIn("--friends", out)

    def test_friends_flag_turns_friends_on(self):
        for argv, expected in ((["goal"], False), (["--friends", "goal"], True)):
            with mock.patch.dict(os.environ, {**env_for(self.tmp), "USE_FRIENDS": "0"}), \
                    mock.patch("agent.app.App") as app_cls:
                cli.main(argv)
            self.assertEqual(app_cls.call_args.args[0].use_friends, expected)

    def test_unknown_flag_is_an_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--frends", "goal"])

    def test_no_goal_exits_with_usage(self):
        with mock.patch.dict(os.environ, env_for(self.tmp)), self.assertRaises(SystemExit) as cm:
            with mock.patch("agent.app.LLM", lambda *a, **k: FakeLLM()):
                cli.main([])
        self.assertIn("usage:", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
