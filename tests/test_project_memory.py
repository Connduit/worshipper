import json
import tempfile
import unittest
from pathlib import Path
from agent.context import compact, total_chars
from agent.storage import ProjectMemory

class ProjectMemoryTests(unittest.TestCase):
    def test_archive_and_recall_after_compaction(self):
        with tempfile.TemporaryDirectory() as d:
            memory = ProjectMemory(str(Path(d) / 'session.json'))
            messages = [{'role':'system','content':'instructions'},
                        {'role':'system','content':'[persistent project memory] earlier'},
                        {'role':'user','content':'GOAL: create a dark theme'},
                        {'role':'assistant','content':'Implemented a dark theme'},
                        {'role':'user','content':'next'}]
            for i in range(10):
                messages.append({'role':'assistant','content':'done ' + 'x'*500})
            compact(messages, 1500, keep_recent=2, on_drop=memory.remember)
            self.assertTrue(Path(memory.archive_path).exists())
            self.assertIn('dark theme', memory.read())
            self.assertEqual(messages[1]['role'], 'system')
            self.assertTrue(all(json.loads(line) for line in Path(memory.archive_path).read_text().splitlines()))

if __name__ == '__main__': unittest.main()
