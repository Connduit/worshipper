"""Server tuning: parsing the current settings, the hard bounds, and the automatic undo."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import FINISHED, Agent                                   # noqa: E402
from agent.console import Console                                         # noqa: E402
from agent.hostcontrol import CTX_MAX, CTX_MIN, Settings, Tuner, parse_unit  # noqa: E402
from agent.selftools import ServerStatusTool, TuneServerTool              # noqa: E402
from agent.storage import EventLog                                        # noqa: E402
from agent.tools import CallContext, FinishTool                           # noqa: E402

LLAMA = "/home/binks/repos/llama.cpp/build/bin/llama-server"


def unit(args: str) -> str:
    return f"[Unit]\nDescription=x\n\n[Service]\nEnvironmentFile=-%h/.config/llama-server.env\nExecStart={LLAMA} {args}\nRestart=on-failure\n"


QWEN = "-hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M --jinja -ngl 99 -c 16384 -np 1 --host 127.0.0.1 --port 8080 -ctk q8_0 -ctv q8_0"
GPTOSS = "-hf ggml-org/gpt-oss-20b-GGUF --jinja -ngl 99 -c 32768 -np 2 --host 127.0.0.1 --port 8080"


class ParseTest(unittest.TestCase):
    def test_a_unit_with_a_compressed_cache(self):
        s = parse_unit(unit(QWEN))
        self.assertEqual(s, Settings("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M", 16384, 1, "q8_0"))

    def test_the_context_is_per_slot(self):
        s = parse_unit(unit(GPTOSS))
        self.assertEqual((s.ctx, s.np, s.total, s.kv_cache), (16384, 2, 32768, "f16"))

    def test_a_local_file_with_spaces_in_its_path(self):
        s = parse_unit(unit('-m "/my models/q.gguf" --jinja -ngl 99 -c 8192 -np 1 --host 127.0.0.1 --port 8080'))
        self.assertEqual(s.model, "/my models/q.gguf")

    def test_other_flags_are_kept_as_they_are(self):
        s = parse_unit(unit(GPTOSS + " --n-cpu-moe 4 --flash-attn -ctk q4_0 -ctv q4_0"))
        self.assertEqual(s.extra, ("--n-cpu-moe", "4", "--flash-attn"))
        self.assertEqual(s.kv_cache, "q4_0")

    def test_unusable_units(self):
        self.assertIsNone(parse_unit("[Service]\nRestart=always\n"))
        self.assertIsNone(parse_unit(unit("--jinja -c 4096")))                       # no model
        self.assertIsNone(parse_unit(unit("-hf a/b --jinja")))                       # no context size
        self.assertEqual(parse_unit(unit(QWEN.replace("q8_0", "weird"))).kv_cache, "f16")


class Proc:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout, self.stderr = rc, out, ""


class TunerBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "deploy").mkdir()
        (self.root / "model.sh").write_text("#!/bin/bash\n")
        self.write_unit(QWEN)
        self.calls, self.results = [], []

    def write_unit(self, args):
        (self.root / "deploy/llama.service").write_text(unit(args))

    def runner(self, argv, env, timeout):
        self.calls.append((argv, env, timeout))
        item = self.results.pop(0) if self.results else Proc(0, "ready")
        if isinstance(item, Exception):
            raise item
        return item

    def tuner(self, **kw):
        return Tuner(self.root, runner=self.runner, gpu=lambda: (20480, 24564), **kw)


class TunerTest(TunerBase):
    def test_a_good_change_restarts_the_server_in_strict_mode(self):
        ok, msg = self.tuner().apply(ctx=32768)
        self.assertTrue(ok)
        self.assertIn("32768 tokens per slot x 1 slot(s) = 32768 total, KV cache q8_0", msg)
        argv, env, _ = self.calls[0]
        self.assertEqual(argv, ["bash", str(self.root / "model.sh"), "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M"])
        self.assertEqual((env["CTX"], env["NP"], env["EXTRA"], env["STRICT"]), ("32768", "1", "-ctk q8_0 -ctv q8_0", "1"))
        self.assertNotIn("NO_CHECK", env)

    def test_switching_the_cache_back_to_f16_and_keeping_other_flags(self):
        self.write_unit(GPTOSS + " --n-cpu-moe 4 -ctk q8_0 -ctv q8_0")
        self.tuner().apply(kv_cache="f16")
        self.assertEqual(self.calls[0][1]["EXTRA"], "--n-cpu-moe 4")
        self.tuner().apply(kv_cache="q4_0")
        self.assertEqual(self.calls[1][1]["EXTRA"], "-ctk q4_0 -ctv q4_0 --n-cpu-moe 4")

    def test_the_callers_environment_cannot_leak_dangerous_switches(self):
        with mock.patch.dict(os.environ, {"DRY_RUN": "1", "NO_CHECK": "1", "SKIP_CHECK": "1", "STRICT": ""}):
            self.tuner().apply(np=2)
        env = self.calls[0][1]
        self.assertEqual([k for k in ("DRY_RUN", "NO_CHECK", "SKIP_CHECK") if k in env], [])
        self.assertEqual(env["STRICT"], "1")

    def test_an_unhealthy_result_is_undone_automatically(self):
        self.results = [Proc(4, "WARNING: the GPU is over 93% full"), Proc(0, "ready")]
        ok, msg = self.tuner().apply(ctx=65536, np=2)
        self.assertFalse(ok)
        self.assertIn("did not work", msg)
        self.assertIn("GPU is over 93% full", msg)
        self.assertIn("restored automatically", msg)
        self.assertEqual(len(self.calls), 2)
        undo_env = self.calls[1][1]
        self.assertEqual((undo_env["CTX"], undo_env["NP"], undo_env["EXTRA"]), ("16384", "1", "-ctk q8_0 -ctv q8_0"))
        self.assertEqual(undo_env["NO_CHECK"], "1")                      # the undo does not run the health check
        self.assertNotIn("STRICT", undo_env)

    def test_a_timeout_counts_as_failure_and_is_undone(self):
        self.results = [subprocess.TimeoutExpired("model.sh", 1000), Proc(0)]
        ok, msg = self.tuner().apply(ctx=32768)
        self.assertFalse(ok)
        self.assertIn("timed out", msg)
        self.assertEqual(len(self.calls), 2)

    def test_if_the_undo_fails_too_it_says_so_loudly(self):
        self.results = [Proc(4, "bad"), Proc(1, "also bad")]
        ok, msg = self.tuner().apply(ctx=32768)
        self.assertFalse(ok)
        self.assertIn("restoring the previous settings failed too", msg)
        self.assertIn("by hand", msg)

    def test_bounds_and_types_are_enforced_before_anything_runs(self):
        t = self.tuner()
        bad = [dict(ctx=CTX_MIN - 1), dict(ctx=CTX_MAX + 1), dict(np=0), dict(np=5), dict(kv_cache="q2_k"),
               dict(kv_cache="f16; rm -rf ~"), dict(ctx="32768"), dict(ctx=True), dict(ctx=1.5), dict()]
        for kw in bad:
            ok, msg = t.apply(**kw)
            self.assertFalse(ok, kw)
        self.assertEqual(self.calls, [])
        self.assertEqual(t.changes_made, 0)

    def test_nothing_to_change_is_not_a_restart(self):
        ok, msg = self.tuner().apply(ctx=16384, np=1, kv_cache="q8_0")
        self.assertTrue(ok)
        self.assertIn("already set", msg)
        self.assertEqual(self.calls, [])

    def test_at_most_three_changes_per_run(self):
        t = self.tuner()
        for ctx in (20000, 24000, 28000):
            self.write_unit(QWEN)                                          # the unit is not rewritten by the fake runner
            self.assertTrue(t.apply(ctx=ctx)[0])
        ok, msg = t.apply(ctx=30000)
        self.assertFalse(ok)
        self.assertIn("limit reached", msg)
        self.assertEqual(len(self.calls), 3)

    def test_missing_pieces_are_reported_clearly(self):
        (self.root / "model.sh").unlink()
        self.assertIn("model.sh was not found", self.tuner().apply(ctx=32768)[1])
        (self.root / "model.sh").write_text("")
        (self.root / "deploy/llama.service").unlink()
        self.assertIn("can't read the current server settings", self.tuner().apply(ctx=32768)[1])

    def test_every_attempt_is_written_to_an_audit_log(self):
        self.results = [Proc(0), Proc(4), Proc(0)]
        t = self.tuner()
        t.apply(ctx=32768)
        t.apply(ctx=65536)
        lines = [json.loads(l) for l in (self.root / ".selfimprove/tune.jsonl").read_text().splitlines()]
        self.assertEqual([e["ok"] for e in lines], [True, False])
        self.assertIn("32768 tokens per slot", lines[0]["to"])

    def test_status_shows_settings_gpu_and_what_may_be_changed(self):
        text = self.tuner().status_text(window=16384)
        self.assertIn("model: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M", text)
        self.assertIn("16384 tokens per slot x 1 slot(s)", text)
        self.assertIn("reports: 16384 tokens", text.replace("as the server reports it: ", "reports: "))
        self.assertIn("GPU memory: 20.0 / 24.0 GB (83% used)", text)
        self.assertIn(f"ctx {CTX_MIN}-{CTX_MAX}", text)
        self.assertIn("changes left this run: 3", text)


class FakeLlm:
    def __init__(self):
        self.forgot = 0

    def context_size(self):
        return 16384

    def forget_window(self):
        self.forgot += 1


class ToolTest(TunerBase):
    def tool(self):
        self.llm = FakeLlm()
        return TuneServerTool(self.tuner(), self.llm)

    def test_success_makes_the_agent_ask_the_server_for_its_window_again(self):
        out = self.tool().run({"ctx": 32768}, CallContext())
        self.assertTrue(out.startswith("OK: "))
        self.assertEqual(self.llm.forgot, 1)

    def test_failure_leaves_the_known_window_alone(self):
        self.results = [Proc(4, "bad"), Proc(0)]
        out = self.tool().run({"ctx": 32768}, CallContext())
        self.assertTrue(out.startswith("FAILED: "))
        self.assertEqual(self.llm.forgot, 0)

    def test_numbers_may_arrive_as_text_but_nothing_else_does(self):
        t = self.tool()
        self.assertTrue(t.run({"ctx": "32768", "np": "1"}, CallContext()).startswith("OK"))
        self.assertEqual(self.calls[0][1]["CTX"], "32768")
        self.assertEqual(t.run({"ctx": "lots"}, CallContext()), "FAILED: ctx must be a whole number.")
        self.assertTrue(t.run({"kv_cache": "q1"}, CallContext()).startswith("FAILED"))

    def test_workers_and_read_only_agents_cannot_use_the_tools(self):
        t = self.tool()
        for ctx in (CallContext(depth=1), CallContext(readonly=True)):
            self.assertIn("only the main agent", t.run({"ctx": 32768}, ctx))
            self.assertIn("only the main agent", ServerStatusTool(self.tuner(), self.llm).run({}, ctx))
        self.assertEqual(self.calls, [])

    def test_status_tool_reports_the_window_the_agent_sees(self):
        out = ServerStatusTool(self.tuner(), FakeLlm()).run({}, CallContext())
        self.assertIn("as the server reports it: 16384 tokens", out)

    def test_the_model_can_do_it_in_a_real_agent_loop(self):
        class Reply:
            def __init__(self, name, **a):
                self.content, self.id = None, f"id-{name}"
                self.tool_calls = [NS(id=self.id, function=NS(name=name, arguments=json.dumps(a)))]

            def model_dump(self, exclude_none=False):
                tc = self.tool_calls[0]
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]}

        class Llm(FakeLlm):
            def __init__(inner, *script):
                super().__init__()
                inner.script = list(script)

            def failure_text(inner):
                return "failed"

            def chat(inner, messages, tools=None, **kw):
                inner.last = messages
                return NS(choices=[NS(message=inner.script.pop(0))])

        llm = Llm(Reply("server_status"), Reply("tune_server", ctx=32768, kv_cache="q8_0"), Reply("finish", answer="done"))
        tuner = self.tuner()
        log = EventLog(f"{self._tmp.name}/log.jsonl")
        agent = Agent("main", llm, [ServerStatusTool(tuner, llm), TuneServerTool(tuner, llm), FinishTool()], 6, log, Console())
        result = agent.run([{"role": "system", "content": "S"}, {"role": "user", "content": "GOAL: you keep running out of context"}])
        self.assertEqual(result.status, FINISHED)
        seen = " ".join(str(m.get("content")) for m in llm.last if m["role"] == "tool")
        self.assertIn("GPU memory: 20.0 / 24.0 GB", seen)                  # it looked first
        self.assertIn("OK: server restarted with 32768 tokens per slot", seen)
        self.assertEqual(llm.forgot, 1)
        self.assertEqual(self.calls[0][1]["CTX"], "32768")


if __name__ == "__main__":
    unittest.main()
