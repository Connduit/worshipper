"""Regression tests for calls rejected before side effects and CLI outcomes."""
import unittest
from unittest import mock
from pathlib import Path

from test_agent import FakeLLM, Msg, TmpCase, call, env_for
from agent import cli
from agent.agent import RunResult, FINISHED, MAX_STEPS, LLM_FAILED, CONTEXT_FULL


class ContractTest(TmpCase):
    def test_invalid_calls_have_no_side_effects(self):
        target = Path(self.tmp) / 'should-not-exist'
        invalid = [
            call('run_shell', command=123),
            call('run_shell', command=f'touch {target}', timeout=True),
            call('run_shell', command=f'touch {target}', timeout=301),
            call('run_shell', command=f'touch {target}', unexpected=True),
            call('write_file', path='bad.txt', content='bad', append='false'),
            call('finish', answer=12),
        ]
        app = self.make_app(FakeLLM(*(Msg(calls=[c]) for c in invalid),
                                    Msg(calls=[call('finish', answer='done')])))
        with mock.patch.object(app.sandbox, 'run') as run:
            result = app.run_goal('test validation')
        run.assert_not_called()
        self.assertFalse(target.exists())
        self.assertFalse((app.cfg.workspace / 'bad.txt').exists())
        self.assertEqual(result.status, FINISHED)
        observations = [m['content'] for m in app.messages if m['role'] == 'tool']
        self.assertTrue(all(o.startswith('ERROR:') for o in observations[:-1]))

    def test_batch_with_finish_and_shell_is_rejected_and_history_answered(self):
        app = self.make_app(FakeLLM(
            Msg(calls=[call('finish', answer='premature'), call('run_shell', command='touch bad')]),
            Msg(calls=[call('finish', answer='done')]),
        ))
        with mock.patch.object(app.sandbox, 'run') as run:
            result = app.run_goal('x')
        run.assert_not_called()
        self.assertEqual(result.text, 'done')
        replies = [m for m in app.messages if m['role'] == 'tool']
        self.assertEqual(len(replies), 3)
        self.assertTrue(all('No calls' in m['content'] for m in replies[:2]))

    def test_cli_propagates_incomplete_outcomes(self):
        for status, code in [(MAX_STEPS, 2), (LLM_FAILED, 3), (CONTEXT_FULL, 4)]:
            with self.subTest(status=status), mock.patch.dict('os.environ', env_for(self.tmp)), \
                    mock.patch('agent.app.App') as app:
                app.return_value.run_goal.return_value = RunResult(status)
                with self.assertRaises(SystemExit) as raised:
                    cli.main(['goal'])
                self.assertEqual(raised.exception.code, code)


if __name__ == '__main__':
    unittest.main()
