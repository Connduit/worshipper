"""Offline tests: no real model downloads or systemd calls."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from agent.model_download import select_files

ROOT = Path(__file__).resolve().parent.parent


class SelectionTest(unittest.TestCase):
    def test_default_quant_and_projector_exclusion(self):
        self.assertEqual(select_files(['m-Q8.gguf', 'm-q4_k_m.gguf', 'mmproj-q4_k_m.gguf'], ''),
                         ['m-q4_k_m.gguf'])

    def test_single_format_shards(self):
        files = ['gpt-MXFP4-00002-of-00002.gguf', 'gpt-MXFP4-00001-of-00002.gguf']
        self.assertEqual(select_files(files, ''), sorted(files))

    def test_ambiguous_or_missing_quant_fails(self):
        for files, quant in [(['m-Q8.gguf', 'm-Q6.gguf'], ''), (['m-Q8.gguf'], 'Q4'), ([], '')]:
            with self.subTest(files=files, quant=quant), self.assertRaises(ValueError):
                select_files(files, quant)


@unittest.skipUnless((ROOT / 'model.sh').is_file(), 'host launcher is not included in the sandbox dev copy')
class DownloadCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.before = (ROOT / 'deploy/llama.service').read_bytes()
        bin_dir = self.path / 'bin'
        bin_dir.mkdir()
        forbidden = bin_dir / 'systemctl'
        forbidden.write_text('#!/bin/sh\necho systemctl-called >> "$HOME/forbidden"\nexit 99\n')
        forbidden.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.path), HF_HOME=str(self.path / 'hf'),
                        LLAMA_CACHE=str(self.path / 'old'), SKIP_CHECK='1',
                        PATH=f'{bin_dir}:{os.environ["PATH"]}')
        for k in ('HF_TOKEN', 'DRY_RUN', 'CTX', 'NP'):
            self.env.pop(k, None)

    def run_script(self, *args, **env):
        result = subprocess.run(['bash', str(ROOT / 'model.sh'), *args], env={**self.env, **env},
                                text=True, capture_output=True, timeout=15)
        self.assertEqual((ROOT / 'deploy/llama.service').read_bytes(), self.before)
        self.assertFalse((self.path / 'forbidden').exists())
        return result

    def test_cached_and_local_files_do_not_contact_services(self):
        cached = self.path / 'hf/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/rev/m-Q4_K_M.gguf'
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b'cached')
        for model in ('qwen-coder', str(cached)):
            result = self.run_script('--download-only', model)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Already available:', result.stdout)

    def test_dry_run_does_not_download(self):
        result = self.run_script('--download-only', 'owner/model:Q8_0', DRY_RUN='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Would download owner/model:Q8_0', result.stdout)

    def test_download_uses_pinned_revision_and_propagates_failure(self):
        module = self.path / 'huggingface_hub.py'
        module.write_text('''import os
from types import SimpleNamespace
class HfApi:
    def __init__(self, **kw): self.endpoint = 'https://huggingface.co'
    def model_info(self, repo, revision): return SimpleNamespace(sha='fixed-revision')
    def list_repo_files(self, repo, revision):
        assert revision == 'fixed-revision'
        return ['model-Q4_K_M.gguf']
def hf_hub_download(**kw):
    assert kw['revision'] == 'fixed-revision'
    if os.environ.get('FAIL_DOWNLOAD'): raise OSError('failure containing secret text')
    return '/cache/model-Q4_K_M.gguf'
''')
        result = self.run_script('--download-only', 'owner/model', PYTHONPATH=str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Download complete', result.stdout)
        result = self.run_script('--download-only', 'owner/model', PYTHONPATH=str(self.path), FAIL_DOWNLOAD='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('secret text', result.stderr)

    def test_missing_model_is_rejected(self):
        self.assertNotEqual(self.run_script('--download-only').returncode, 0)


if __name__ == '__main__':
    unittest.main()
