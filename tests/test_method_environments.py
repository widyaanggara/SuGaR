from pathlib import Path
from types import SimpleNamespace
import unittest
import subprocess
from unittest.mock import patch

from pipeline_tools.check_method_environments import check, method_entrypoints, require_training_environment


class MethodEnvironmentTests(unittest.TestCase):
    def test_missing_rasterizer_reports_selected_interpreter(self):
        failure = subprocess.CalledProcessError(1, ['python'],
            stderr="ModuleNotFoundError: No module named 'diff_surfel_rasterization'")
        with patch.object(Path, 'is_file', return_value=True), \
                patch('pipeline_tools.check_method_environments.subprocess.run', side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, 'Python: /env/sugar/bin/python') as error:
                require_training_environment('2dgs', '/env/sugar/bin/python', Path('/repo'))
        self.assertIn('diff_surfel_rasterization', str(error.exception))

    def test_preflight_uses_shared_launcher_without_training(self):
        with patch.object(Path, 'is_file', return_value=True), \
                patch('pipeline_tools.check_method_environments.subprocess.run') as probe:
            require_training_environment('2dgs', '/env/2dgs/python', Path('/repo'), 2)
        command = probe.call_args.args[0]
        self.assertEqual(command[0], '/env/2dgs/python')
        self.assertTrue(command[1].endswith('external_visual_launcher.py'))
        self.assertEqual(command[-2:], ['train.py', '--help'])
        self.assertEqual(probe.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')

    def test_launcher_failure_fails_preflight_even_when_native_probes_pass(self):
        for metric in ('metrics.py', 'metric.py'):
            with self.subTest(metric=metric):
                repo = Path('/external/repo')
                entries = method_entrypoints(repo, ('train.py', 'render.py', metric))
                def run(command, **kwargs):
                    wrapped = any('external_visual_launcher.py' in arg for arg in command)
                    return SimpleNamespace(returncode=int(wrapped), stdout='', stderr='import failure' if wrapped else '')
                with patch.object(Path, 'is_file', return_value=True), \
                        patch('pipeline_tools.check_method_environments.subprocess.run', side_effect=run) as probe, \
                        patch('builtins.print'):
                    self.assertFalse(check('method', 'python', repo, entries, 0))
                self.assertEqual(probe.call_count, 7)
                shared = [call.args[0] for call in probe.call_args_list
                          if any('external_visual_launcher.py' in arg for arg in call.args[0])]
                self.assertEqual([command[-2:] for command in shared],
                                 [[script, '--help'] for script in ('train.py', 'render.py', metric)])


if __name__ == '__main__':
    unittest.main()
