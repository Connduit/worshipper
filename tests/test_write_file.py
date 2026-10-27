"""write_file: creating and extending files in the workspace, and refusing to leave it."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.tools import CallContext, WriteFileTool  # noqa: E402


class WriteFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ws = Path(self._tmp.name) / "workspace"
        self.ws.mkdir()
        self.outside = Path(self._tmp.name) / "outside"
        self.outside.mkdir()
        self.tool = WriteFileTool(self.ws)

    def write(self, path, content, **kw):
        return self.tool.run({"path": path, "content": content, **kw}, CallContext())

    def test_creates_a_file_and_parent_directories(self):
        out = self.write("src/app/main.js", "let a = 1;\nlet b = 2;\n")
        self.assertEqual((self.ws / "src/app/main.js").read_text(), "let a = 1;\nlet b = 2;\n")
        self.assertIn("wrote 22 bytes (2 lines) to src/app/main.js", out)

    def test_overwrites_by_default(self):
        self.write("a.txt", "old old old")
        self.write("a.txt", "new")
        self.assertEqual((self.ws / "a.txt").read_text(), "new")

    def test_append_builds_a_big_file_in_pieces(self):
        self.write("big.txt", "part 1\n")
        out = self.write("big.txt", "part 2\n", append=True)
        self.write("big.txt", "part 3\n", append=True)
        self.assertEqual((self.ws / "big.txt").read_text(), "part 1\npart 2\npart 3\n")
        self.assertIn("appended 7 bytes", out)
        self.assertIn("now 14 bytes", out)

    def test_append_creates_a_missing_file(self):
        self.write("new.txt", "x", append=True)
        self.assertEqual((self.ws / "new.txt").read_text(), "x")

    def test_the_workspace_prefix_the_model_sees_is_accepted(self):
        self.write("/workspace/notes/todo.txt", "hi")
        self.assertEqual((self.ws / "notes/todo.txt").read_text(), "hi")

    def test_unicode_and_a_missing_trailing_newline(self):
        out = self.write("u.txt", "héllo\nwörld")
        self.assertEqual((self.ws / "u.txt").read_text(encoding="utf-8"), "héllo\nwörld")
        self.assertIn("(2 lines)", out)

    def test_refuses_paths_outside_the_workspace(self):
        for bad in ("../outside/evil.txt", "a/../../outside/evil.txt", "/etc/passwd", "/tmp/x"):
            self.assertTrue(self.write(bad, "x").startswith("ERROR"), bad)
        self.assertEqual(list(self.outside.iterdir()), [])

    def test_refuses_to_write_through_a_symlink_that_leaves_the_workspace(self):
        (self.outside / "secret.txt").write_text("keep me")
        os.symlink(self.outside / "secret.txt", self.ws / "link.txt")        # as a sandboxed command could
        os.symlink(self.outside, self.ws / "linkdir")
        self.assertTrue(self.write("link.txt", "overwritten").startswith("ERROR"))
        self.assertTrue(self.write("linkdir/new.txt", "x").startswith("ERROR"))
        self.assertEqual((self.outside / "secret.txt").read_text(), "keep me")
        self.assertFalse((self.outside / "new.txt").exists())

    def test_a_symlink_that_stays_inside_is_fine(self):
        (self.ws / "real").mkdir()
        os.symlink(self.ws / "real", self.ws / "alias")
        self.write("alias/f.txt", "ok")
        self.assertEqual((self.ws / "real/f.txt").read_text(), "ok")

    def test_directories_empty_paths_and_bad_content_are_errors(self):
        (self.ws / "d").mkdir()
        self.assertIn("directory", self.write("d", "x"))
        self.assertIn("give the path", self.write("", "x"))
        self.assertIn("give the path", self.write(".", "x"))
        self.assertIn("must be a string", self.write("a.txt", {"not": "text"}))
        self.assertFalse((self.ws / "a.txt").exists())

    def test_oversized_content_is_refused_with_advice(self):
        out = self.write("big.txt", "x" * (WriteFileTool.MAX_BYTES + 1))
        self.assertIn("append=true", out)
        self.assertFalse((self.ws / "big.txt").exists())

    def test_read_only_workers_cannot_write(self):
        out = self.tool.run({"path": "a.txt", "content": "x"}, CallContext(readonly=True))
        self.assertIn("read-only", out)
        self.assertFalse((self.ws / "a.txt").exists())

    def test_missing_arguments_raise_keyerror_for_the_agent_to_report(self):
        with self.assertRaises(KeyError):
            self.tool.run({"content": "x"}, CallContext())


if __name__ == "__main__":
    unittest.main()
