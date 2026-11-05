"""The promotion gate: what may be promoted, what is verified, and that every failure is undone."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import selfimprove  # noqa: E402
from agent.selfimprove import Promoter, risk_increase  # noqa: E402

UTIL_OK = "def add(a, b):\n    return a + b\n"
TEST_OK = ("import sys, unittest\nfrom pathlib import Path\n"
           "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n"
           "from agent.util import add\n\n\nclass T(unittest.TestCase):\n"
           "    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n")
ENTRY = "import sys\nif '-h' in sys.argv:\n    print('usage')\n"


def make_repo(base: Path) -> tuple[Path, Path]:
    root = base / "code"
    for rel, text in {"agent_local.py": ENTRY, "agent/__init__.py": "", "agent/util.py": UTIL_OK,
                      "agent/llm.py": "X = 1\n", "agent/sandbox.py": "JAIL = 'strict'\n",
                      "tests/test_util.py": TEST_OK}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    dev = base / "workspace" / "agent_dev"
    shutil.copytree(root, dev)
    return root, dev


class LocalRunner:
    """Stands in for the jail: runs the verification commands on a temp copy of the layout."""
    def __init__(self, root, dev):
        self.root, self.dev, self.calls = root, dev, []

    def __call__(self, command, timeout, binds):
        self.calls.append((command, binds))
        cmd = (command.replace("/workspace/agent_dev", str(self.dev)).replace("/repo", str(self.root))
               .replace("python ", f"{sys.executable} "))
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.root, self.dev = make_repo(self.base)
        self.runner = LocalRunner(self.root, self.dev)

    def promoter(self, runner=None, **kw):
        return Promoter(self.root, self.dev, runner or self.runner, state_dir=self.base / "state", **kw)

    def edit(self, rel, text):
        (self.dev / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.dev / rel).write_text(text)

    def host(self, rel):
        return (self.root / rel).read_text()


class RiskCheckTest(unittest.TestCase):
    def test_a_harmless_change_adds_nothing_risky(self):
        self.assertEqual(risk_increase("def f():\n    return 1\n", "def f():\n    return 2  # better\n"), [])

    def test_new_risky_capabilities_are_caught(self):
        cases = {"import subprocess\n": "import subprocess", "from socket import socket\n": "import socket",
                 "import os\nos.system('x')\n": "os.system", "eval('1')\n": "eval()", "exec('1')\n": "exec()",
                 "open('f', 'w')\n": "open()", "x = ().__class__.__bases__\n": "__class__",
                 "import os\nos.environ['A']\n": "os.environ", "__import__('os')\n": "__import__()"}
        for src, expect in cases.items():
            reasons = risk_increase("", src)
            self.assertTrue(any(expect in r for r in reasons), (src, reasons))

    def test_existing_risky_use_is_not_counted_again_only_additions_are(self):
        old = "import subprocess\nsubprocess.run(['ls'])\nopen('a')\n"
        self.assertEqual(risk_increase(old, old + "\n# comment\n"), [])
        self.assertEqual(risk_increase(old, old.replace("open('a')", "open('b')")), [])
        more = risk_increase(old, old + "open('c')\n")
        self.assertEqual(len(more), 1)
        self.assertIn("open()", more[0])

    def test_syntax_errors_raise(self):
        with self.assertRaises(SyntaxError):
            risk_increase("", "def broken(:\n")


class ClassifyTest(Base):
    def verdicts(self):
        return {c.path: c.verdict for c in self.promoter().changes()}

    def test_no_differences(self):
        self.assertEqual(self.promoter().changes(), [])

    def test_each_kind_of_change_is_judged(self):
        self.edit("agent/util.py", UTIL_OK + "\n# tweak\n")                       # fine
        self.edit("agent/sandbox.py", "JAIL = 'open'\n")                          # protected
        self.edit("agent/llm.py", "import subprocess\nsubprocess.run(['ls'])\n")   # new capability
        self.edit("agent/new_helper.py", "def h():\n    return 1\n")               # added, fine
        self.edit("agent/oops.py", "def broken(:\n")                              # unparsable
        (self.dev / "tests/test_util.py").unlink()                                # deleted
        v = self.verdicts()
        self.assertEqual(v, {"agent/util.py": "auto", "agent/sandbox.py": "protected", "agent/llm.py": "risky",
                             "agent/new_helper.py": "auto", "agent/oops.py": "unparsable",
                             "tests/test_util.py": "deleted"})

    def test_the_gate_and_the_host_side_tools_are_protected(self):
        for rel in ("agent_local.py", "agent/__init__.py", "agent/cli.py", "agent/app.py", "agent/config.py",
                    "agent/sandbox.py", "agent/tools.py", "agent/devcopy.py", "agent/selfimprove.py",
                    "agent/selftools.py", "agent/hostcontrol.py"):
            self.assertIn(rel, selfimprove.PROTECTED)

    def test_constants_match_the_rest_of_the_code(self):
        from agent import config, devcopy
        self.assertEqual(selfimprove.DEV_NAME, config.DEV_NAME)
        self.assertEqual(selfimprove.ITEMS, devcopy.ITEMS)


class PromoteTest(Base):
    def test_a_good_change_is_promoted_verified_backed_up_and_logged(self):
        self.edit("agent/util.py", "def add(a, b):\n    return a + b  # improved\n")
        res = self.promoter().promote(auto=True)
        self.assertEqual(res.applied, ["agent/util.py"])
        self.assertEqual(res.failed, "")
        self.assertIn("improved", self.host("agent/util.py"))
        cmds = [c for c, _ in self.runner.calls]
        self.assertEqual(len(cmds), 2)                                    # dev tests, then the new real tree
        self.assertIn("agent_dev", cmds[0])
        self.assertEqual(self.runner.calls[1][1], ((str(self.root), "/repo"),))   # real code shown to the jail read-only
        log = [json.loads(line) for line in (self.base / "state/log.jsonl").read_text().splitlines()]
        self.assertEqual(log[0]["files"], ["agent/util.py"])
        self.assertIn("improved", log[0]["diff"])
        self.assertTrue(list((self.base / "state/backups").glob("*/files/agent/util.py")))
        self.assertEqual(self.promoter().changes(), [])                  # nothing left to promote

    def test_failing_tests_in_the_dev_copy_stop_everything_before_any_file_is_touched(self):
        self.edit("agent/util.py", "def add(a, b):\n    return a - b\n")      # breaks test_add
        res = self.promoter().promote(auto=True)
        self.assertEqual(res.applied, [])
        self.assertIn("failed in the dev copy", res.failed)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)
        self.assertEqual(len(self.runner.calls), 1)

    def test_a_change_that_breaks_the_real_tree_is_rolled_back_and_never_retried(self):
        # util changes behaviour and the dev test agrees, but the test change is held back by a
        # human-review rule, so the real tree's old test fails against the new util
        self.edit("agent/util.py", "def add(a, b):\n    return a + b + 1\n")
        self.edit("tests/test_util.py", TEST_OK.replace("add(1, 2), 3", "add(1, 2), 4"))
        p = self.promoter(protected=selfimprove.PROTECTED | {"tests/test_util.py"})
        res = p.promote(auto=True)
        self.assertEqual(res.applied, [])
        self.assertIn("rolled back", res.failed)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)            # the old file is back
        verdicts = {c.path: c.verdict for c in p.changes()}
        self.assertEqual(verdicts["agent/util.py"], "rejected")
        calls = len(self.runner.calls)
        res2 = p.promote(auto=True)                                       # the model tries the same thing again
        self.assertEqual(res2.applied, [])
        self.assertEqual(len(self.runner.calls), calls)                   # and nothing is even run

    def test_a_timeout_in_the_final_check_counts_as_failure_and_rolls_back(self):
        self.edit("agent/util.py", UTIL_OK + "# x\n")
        n = {"i": 0}
        real = self.runner

        def flaky(command, timeout, binds):
            n["i"] += 1
            if n["i"] == 2:
                raise subprocess.TimeoutExpired(command, timeout)
            return real(command, timeout, binds)

        res = self.promoter(flaky).promote(auto=True)
        self.assertEqual(res.applied, [])
        self.assertIn("timed out", res.failed)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_an_unexpected_error_after_applying_still_restores_the_old_files(self):
        self.edit("agent/util.py", UTIL_OK + "# x\n")
        n = {"i": 0}
        real = self.runner

        def boom(command, timeout, binds):
            n["i"] += 1
            if n["i"] == 2:
                raise KeyboardInterrupt()
            return real(command, timeout, binds)

        with self.assertRaises(KeyboardInterrupt):
            self.promoter(boom).promote(auto=True)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_protected_and_risky_changes_are_queued_not_applied(self):
        self.edit("agent/util.py", UTIL_OK + "# fine\n")
        self.edit("agent/sandbox.py", "JAIL = 'open'\n")
        self.edit("agent/llm.py", "import subprocess\nsubprocess.run(['ls'])\n")
        res = self.promoter().promote(auto=True)
        self.assertEqual(res.applied, ["agent/util.py"])
        queued = dict(res.queued)
        self.assertEqual(set(queued), {"agent/sandbox.py", "agent/llm.py"})
        self.assertIn("protected", queued["agent/sandbox.py"])
        self.assertIn("subprocess", queued["agent/llm.py"])
        self.assertEqual(self.host("agent/sandbox.py"), "JAIL = 'strict'\n")
        self.assertEqual(self.host("agent/llm.py"), "X = 1\n")
        pending = json.loads((self.base / "state/pending.json").read_text())
        self.assertEqual({p["path"] for p in pending}, set(queued))
        self.assertIn("--promote", res.text())

    def test_a_file_that_does_not_parse_blocks_the_whole_promotion(self):
        self.edit("agent/util.py", UTIL_OK + "# fine\n")
        self.edit("agent/oops.py", "def broken(:\n")
        res = self.promoter().promote(auto=True)
        self.assertEqual(res.applied, [])                                 # compileall fails in the dev copy
        self.assertIn("failed in the dev copy", res.failed)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)
        self.assertFalse((self.root / "agent/oops.py").exists())

    def test_queue_mode_applies_nothing(self):
        self.edit("agent/util.py", UTIL_OK + "# fine\n")
        res = self.promoter().promote(auto=False)
        self.assertEqual(res.applied, [])
        self.assertEqual(self.runner.calls, [])
        self.assertEqual([p for p, _ in res.queued], ["agent/util.py"])
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_too_many_files_are_left_for_a_human(self):
        for i in range(selfimprove.MAX_AUTO_FILES + 1):
            self.edit(f"agent/mod{i}.py", f"V = {i}\n")
        res = self.promoter().promote(auto=True)
        self.assertEqual(res.applied, [])
        self.assertTrue(all("too big" in why for _, why in res.queued))

    def test_nothing_to_do(self):
        res = self.promoter().promote(auto=True)
        self.assertTrue(res.nothing)
        self.assertIn("No changes", res.text())

    def test_a_human_can_approve_protected_and_risky_changes(self):
        self.edit("agent/sandbox.py", "JAIL = 'reviewed'\n")
        self.edit("agent/llm.py", "import subprocess\nsubprocess.run(['ls'])\n")
        res = self.promoter().promote(auto=False, approve_all=True)
        self.assertEqual(sorted(res.applied), ["agent/llm.py", "agent/sandbox.py"])
        self.assertEqual(self.host("agent/sandbox.py"), "JAIL = 'reviewed'\n")
        self.assertEqual(len(self.runner.calls), 2)                       # approval does not skip verification


class RollbackTest(Base):
    def test_rollback_restores_modified_files_and_removes_added_ones(self):
        self.edit("agent/util.py", UTIL_OK + "# v2\n")
        self.edit("agent/extra.py", "E = 1\n")
        p = self.promoter()
        p.promote(auto=True)
        self.assertTrue((self.root / "agent/extra.py").exists())
        restored = p.rollback()
        self.assertEqual(sorted(restored), ["agent/extra.py", "agent/util.py"])
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)
        self.assertFalse((self.root / "agent/extra.py").exists())
        self.assertEqual(p.rollback(), [])                                # nothing left to undo

    def test_a_rolled_back_change_is_not_promoted_again_automatically(self):
        self.edit("agent/util.py", UTIL_OK + "# v2\n")
        p = self.promoter()
        p.promote(auto=True)
        p.rollback()
        self.assertEqual(p.changes()[0].verdict, "rejected")
        self.assertEqual(p.promote(auto=True).applied, [])

    def test_rollback_undoes_only_the_latest_promotion(self):
        p = self.promoter()
        self.edit("agent/util.py", UTIL_OK + "# first\n")
        p.promote(auto=True)
        self.edit("agent/util.py", UTIL_OK + "# first\n# second\n")
        p.promote(auto=True)
        p.rollback()
        self.assertIn("# first", self.host("agent/util.py"))
        self.assertNotIn("# second", self.host("agent/util.py"))

    def test_a_missing_dev_copy_is_a_clear_error(self):
        shutil.rmtree(self.dev)
        with self.assertRaises(FileNotFoundError):
            self.promoter().promote(auto=True)


class SelftestTest(Base):
    def test_a_healthy_tree_passes_both_checks_and_nothing_changes(self):
        ok, text = self.promoter().selftest()
        self.assertTrue(ok)
        self.assertIn("real code (tests and -h, in the jail): ok", text)
        self.assertIn("dev copy (compile and tests, in the jail): ok", text)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)
        self.assertFalse((self.base / "state/log.jsonl").exists())

    def test_a_broken_real_tree_fails_with_the_output(self):
        (self.root / "agent/util.py").write_text("def add(a, b):\n    return 0\n")
        ok, text = self.promoter().selftest()
        self.assertFalse(ok)
        self.assertIn("real code (tests and -h, in the jail): FAILED", text)
        self.assertIn("AssertionError", text)

    def test_without_a_dev_copy_only_the_real_code_is_checked(self):
        shutil.rmtree(self.dev)
        ok, text = self.promoter().selftest()
        self.assertTrue(ok)
        self.assertNotIn("dev copy", text)


class ReviewTest(Base):
    def review(self, answer="y", interactive=True):
        out, asked = [], []
        code = selfimprove.review_and_promote(self.promoter(), ask=lambda q: (asked.append(q), answer)[1],
                                              out=out.append, interactive=interactive)
        return code, "\n".join(out), asked

    def test_shows_each_change_with_its_verdict_and_the_diff_then_applies_on_yes(self):
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        self.edit("agent/sandbox.py", "JAIL = 'reviewed'\n")
        code, out, asked = self.review("y")
        self.assertEqual(code, 0)
        self.assertIn("2 changed file(s):", out)
        self.assertRegex(out, r"auto\s+modified\s+agent/util.py")
        self.assertRegex(out, r"protected\s+modified\s+agent/sandbox.py")
        self.assertIn("+# tweak", out)
        self.assertIn("+JAIL = 'reviewed'", out)
        self.assertIn("Verify in the jail, then apply 2 change(s)?", asked[0])
        self.assertIn("PROMOTED", out)
        self.assertEqual(self.host("agent/sandbox.py"), "JAIL = 'reviewed'\n")      # a human may approve a protected file

    def test_anything_but_yes_applies_nothing(self):
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        for answer in ("n", "", "yes please maybe", "N"):
            code, out, _ = self.review(answer)
            self.assertEqual(code, 0)
            self.assertIn("Nothing applied.", out)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)
        self.assertEqual(self.runner.calls, [])

    def test_without_a_terminal_it_refuses_to_apply(self):
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        code, out, asked = self.review(interactive=False)
        self.assertEqual(code, 1)
        self.assertIn("needs a terminal", out)
        self.assertEqual(asked, [])
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_no_differences(self):
        code, out, asked = self.review()
        self.assertEqual((code, out, asked), (0, "No differences.", []))

    def test_a_file_that_does_not_parse_is_reported_and_not_counted(self):
        self.edit("agent/oops.py", "def broken(:\n")
        code, out, asked = self.review("y")
        self.assertIn("Not applied, they do not parse: agent/oops.py", out)
        self.assertEqual(asked, [])                                    # nothing left to ask about
        self.assertFalse((self.root / "agent/oops.py").exists())

    def test_a_failed_promotion_exits_nonzero(self):
        self.edit("agent/util.py", "def add(a, b):\n    return a - b\n")           # breaks the dev tests
        code, out, _ = self.review("y")
        self.assertEqual(code, 1)
        self.assertIn("NOT PROMOTED", out)


def _reply(name, **args):
    from types import SimpleNamespace as NS
    tc = NS(id=f"id-{name}", function=NS(name=name, arguments=json.dumps(args)))
    msg = NS(content=None, tool_calls=[tc])
    msg.model_dump = lambda exclude_none=False: {"role": "assistant", "content": None, "tool_calls": [
        {"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]}
    return msg


class ToolTest(Base):
    def tool(self, auto=True, **kw):
        from agent.selftools import PromoteChangesTool
        return PromoteChangesTool(self.promoter(), auto=auto, **kw)

    def run_tool(self, tool, ctx=None):
        from agent.tools import CallContext
        return tool.run({}, ctx or CallContext())

    def test_in_auto_mode_a_good_change_is_promoted(self):
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        out = self.run_tool(self.tool())
        self.assertIn("PROMOTED (active from the next start): agent/util.py", out)
        self.assertIn("# tweak", self.host("agent/util.py"))

    def test_in_queue_mode_nothing_is_applied_and_the_model_is_told_how_a_human_can(self):
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        out = self.run_tool(self.tool(auto=False))
        self.assertNotIn("PROMOTED", out)
        self.assertIn("--promote", out)
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_protected_changes_are_explained_to_the_model(self):
        self.edit("agent/sandbox.py", "JAIL = 'open'\n")
        out = self.run_tool(self.tool())
        self.assertIn("agent/sandbox.py: protected", out)
        self.assertEqual(self.host("agent/sandbox.py"), "JAIL = 'strict'\n")

    def test_workers_and_read_only_agents_cannot_promote(self):
        from agent.tools import CallContext
        self.edit("agent/util.py", UTIL_OK + "# tweak\n")
        t = self.tool()
        for ctx in (CallContext(depth=1), CallContext(readonly=True)):
            self.assertIn("only the main agent", self.run_tool(t, ctx))
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_at_most_three_per_run(self):
        t = self.tool()
        for _ in range(3):
            self.run_tool(t)
        self.assertIn("limit reached", self.run_tool(t))

    def test_a_missing_dev_copy_is_a_readable_error_not_a_crash(self):
        shutil.rmtree(self.dev)
        self.assertTrue(self.run_tool(self.tool()).startswith("ERROR: No dev copy"))

    def test_the_model_can_do_it_in_a_real_agent_loop(self):
        from types import SimpleNamespace as NS
        from agent.agent import FINISHED, Agent
        from agent.console import Console
        from agent.storage import EventLog
        from agent.tools import FinishTool

        class Llm:
            def __init__(inner, *script):
                inner.script = list(script)

            def context_size(inner):
                return None

            def failure_text(inner):
                return "failed"

            def chat(inner, messages, tools=None, **kw):
                inner.last = messages
                return NS(choices=[NS(message=inner.script.pop(0))])

        self.edit("agent/util.py", "def add(a, b):\n    return a + b  # the model's improvement\n")
        self.edit("agent/sandbox.py", "JAIL = 'weakened'\n")                     # and one change it may not make alone
        llm = Llm(_reply("promote_changes"), _reply("finish", answer="done"))
        agent = Agent("main", llm, [self.tool(), FinishTool()], 5, EventLog(str(self.base / "log.jsonl")), Console())
        result = agent.run([{"role": "system", "content": "S"}, {"role": "user", "content": "GOAL: improve yourself"}])
        self.assertEqual(result.status, FINISHED)
        seen = next(m["content"] for m in llm.last if m["role"] == "tool")
        self.assertIn("PROMOTED (active from the next start): agent/util.py", seen)
        self.assertIn("agent/sandbox.py: protected", seen)                        # it learns what needs a human
        self.assertIn("improvement", self.host("agent/util.py"))
        self.assertEqual(self.host("agent/sandbox.py"), "JAIL = 'strict'\n")      # and the jail definition is untouched


class CliTest(unittest.TestCase):
    def test_the_new_flags_parse(self):
        from agent.cli import build_parser
        a = build_parser().parse_args(["--promote", "--rollback", "--verify"])
        self.assertTrue(a.promote and a.rollback and a.verify)

    def run_command(self, flag, sandbox_mode="bwrap", **fake):
        from agent import cli
        from agent.config import Config
        cfg = Config.from_env({"SANDBOX_MODE": sandbox_mode, "WORKSPACE": "/tmp/ws-cli-test"})
        args = cli.build_parser().parse_args([flag])
        promoter = mock.Mock(**fake)
        out = io.StringIO()
        with mock.patch("agent.sandbox.BwrapSandbox"), mock.patch("agent.selfimprove.Promoter", return_value=promoter), \
                mock.patch("agent.selfimprove.review_and_promote", return_value=7) as review, contextlib.redirect_stdout(out):
            code = cli._self_improvement_command(args, cfg)
        return code, out.getvalue(), promoter, review

    def test_they_refuse_to_run_without_the_jail(self):
        for mode in ("none", "docker"):
            code, out, promoter, _ = self.run_command("--promote", sandbox_mode=mode)
            self.assertEqual(code, 1)
            self.assertIn("SANDBOX_MODE=bwrap", out)
            promoter.changes.assert_not_called()

    def test_rollback_reports_what_was_restored(self):
        code, out, _, _ = self.run_command("--rollback", **{"rollback.return_value": ["agent/llm.py", "agent/agent.py"]})
        self.assertEqual((code, out.strip()), (0, "Restored: agent/llm.py, agent/agent.py"))
        code, out, _, _ = self.run_command("--rollback", **{"rollback.return_value": []})
        self.assertEqual(out.strip(), "Nothing to roll back.")

    def test_verify_exit_code_follows_the_result(self):
        code, out, _, _ = self.run_command("--verify", **{"selftest.return_value": (True, "all ok")})
        self.assertEqual((code, out.strip()), (0, "all ok"))
        code, out, _, _ = self.run_command("--verify", **{"selftest.return_value": (False, "real code: FAILED")})
        self.assertEqual(code, 1)

    def test_promote_hands_over_to_the_review_flow(self):
        code, _, _, review = self.run_command("--promote")
        self.assertEqual(code, 7)
        review.assert_called_once()


class ConfigAndPromptTest(unittest.TestCase):
    def test_modes_and_defaults(self):
        from agent.config import Config
        c = Config.from_env({})
        self.assertEqual((c.self_improve, c.tune_server), ("queue", False))
        for mode in ("off", "queue", "auto", "AUTO"):
            self.assertEqual(Config.from_env({"SELF_IMPROVE": mode}).self_improve, mode.lower())
        self.assertTrue(Config.from_env({"TUNE_SERVER": "1"}).tune_server)
        with self.assertRaises(ValueError):
            Config.from_env({"SELF_IMPROVE": "yolo"})

    def test_the_prompt_only_describes_what_exists(self):
        from agent.config import Config
        from agent.friends import FriendRegistry
        from agent.prompts import build_system

        def prompt(**env):
            return build_system(Config.from_env(env), FriendRegistry({}))

        auto = prompt(SANDBOX_MODE="bwrap", SELF_IMPROVE="auto", TUNE_SERVER="1")
        self.assertIn("promote_changes", auto)
        self.assertIn("applied for you", auto)
        self.assertIn("tune_server", auto)
        queue = prompt(SANDBOX_MODE="bwrap")
        self.assertIn("--promote", queue)
        self.assertNotIn("applied for you", queue)
        self.assertNotIn("tune_server", queue)
        self.assertNotIn("promote_changes", prompt(SANDBOX_MODE="bwrap", SELF_IMPROVE="off"))
        self.assertNotIn("promote_changes", prompt(SANDBOX_MODE="none"))          # no jail, no self-improvement


class SandboxBindTest(unittest.TestCase):
    """Promotion shows the real code to the jail read-only; the exact bwrap arguments matter."""

    def argv(self, **kw):
        from agent.sandbox import BwrapSandbox
        return BwrapSandbox("/tmp/ws").command("echo hi", **kw)

    def test_extra_paths_are_bound_read_only_at_the_given_place(self):
        argv = self.argv(ro_extra=(("/home/me/code", "/repo"),))
        i = argv.index("/home/me/code")
        self.assertEqual(argv[i - 1: i + 2], ["--ro-bind", "/home/me/code", "/repo"])
        self.assertLess(i, argv.index("--chdir"))                      # before the command starts
        self.assertNotIn("--share-net", argv)                          # still no network

    def test_the_default_jail_is_unchanged(self):
        argv = self.argv()
        self.assertNotIn("/repo", argv)
        self.assertIn("--unshare-all", argv)
        self.assertEqual(argv[-3:-1], ["bash", "-c"])

    def test_only_the_workspace_is_writable_even_with_extras(self):
        argv = self.argv(ro_extra=(("/home/me/code", "/repo"),))
        writable = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
        self.assertEqual(writable, ["/tmp/ws"])


class StandaloneCommandsTest(Base):
    """`python -m agent.selfimprove`: must work with nothing but the standard library."""

    def run_main(self, *argv):
        out = io.StringIO()
        with mock.patch.object(selfimprove, "ROOT", self.root), \
                mock.patch.dict("os.environ", {"WORKSPACE": str(self.base / "workspace")}), \
                contextlib.redirect_stdout(out):
            code = selfimprove.main(list(argv))
        return code, out.getvalue().strip()

    def test_status_and_rollback_when_nothing_happened(self):
        self.assertEqual(self.run_main("status"), (0, "No promotions yet."))
        self.assertEqual(self.run_main("rollback"), (0, "Nothing to roll back."))

    def test_status_lists_promotions_and_rollback_undoes_the_last_one(self):
        self.edit("agent/util.py", UTIL_OK + "# v2\n")
        Promoter(self.root, self.dev, self.runner).promote(auto=True)      # state goes to <root>/.selfimprove
        code, out = self.run_main("status")
        self.assertEqual(code, 0)
        self.assertIn("promoted: agent/util.py", out)
        self.assertEqual(self.run_main("rollback"), (0, "Restored: agent/util.py"))
        self.assertEqual(self.host("agent/util.py"), UTIL_OK)

    def test_unknown_command_prints_usage(self):
        code, out = self.run_main("explode")
        self.assertEqual(code, 2)
        self.assertIn("usage:", out)


if __name__ == "__main__":
    unittest.main()
