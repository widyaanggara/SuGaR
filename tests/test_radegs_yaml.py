import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pipeline_tools import run_external_method as runner
from pipeline_tools.radegs_yaml import build_plan
from pipeline_tools.external_methods import checkpoint, mesh_path
from web_dashboard.server import prepare_jobs, build_command

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'external/RaDe-GS'


@unittest.skipUnless(importlib.util.find_spec('yaml') and (REPO / 'research_pipeline/config.py').is_file(),
                     'Requires PyYAML and native RaDe-GS configuration modules')
class RadegsYamlTests(unittest.TestCase):
    def test_both_native_profiles_build_expected_commands_without_donor_paths(self):
        for name in ('togog_bali.yaml', 'tanks_temples.yaml'):
            scene, output = ROOT / 'datasets/test_scene', ROOT / 'outputs/test_scene'
            plan = build_plan(REPO, ROOT / 'pipeline_tools/configs/radegs' / name, scene, output)
            train = plan['commands']['training']
            self.assertIn('--enable_allocator_gc', train)
            self.assertEqual(train[train.index('--resolution') + 1], '1920')
            self.assertEqual(train[train.index('-s') + 1], str(scene))
            self.assertEqual(train[train.index('-m') + 1], str(output))
            self.assertNotIn('geometry_evaluation', plan['commands'])
            self.assertTrue(plan['config']['resource_guard']['enabled'])
            self.assertEqual(plan['config']['resource_guard']['max_wait_seconds'], 300)
            self.assertEqual(plan['commands']['mesh_extraction'][-6:],
                             ['-r', '2', '--num_cluster', '8', '--depth_ratio', '0.5'])

    def test_dataset_selection_execution_guard_and_changed_yaml_resume(self):
        for cls, model, name in (('Dewa', 'model_01', 'togog_bali.yaml'),
                                 ('TanksAndTemples', 'tanks_and_temples_1', 'tanks_temples.yaml')):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                scene = root / 'datasets' / cls / model
                output = root / 'outputs/radegs' / cls / model
                repo = root / 'external/RaDe-GS'
                repo.mkdir(parents=True)
                (repo / 'train.py').write_text('')
                plan = build_plan(REPO, ROOT / 'pipeline_tools/configs/radegs' / name, scene, output)
                args = argparse.Namespace(statue_class=cls, model_id=model, method='radegs',
                    python=sys.executable, gpu=0, resolution=1, radegs_profile='yaml',
                    evaluate_only=False, no_resume=False)
                calls = []
                def subprocess_run(command, **kwargs):
                    calls.append(command)
                    if '--config' in command:
                        self.assertEqual(Path(command[command.index('--config') + 1]).name, name)
                    return argparse.Namespace(stdout=json.dumps(plan))
                process = MagicMock()
                process.poll.return_value = 0
                def finish():
                    checkpoint(output).parent.mkdir(parents=True, exist_ok=True)
                    checkpoint(output).write_text('trained')
                    (output / 'cfg_args').write_text('config')
                    (output / 'results.json').write_text(json.dumps(
                        {'ours_30000': {'PSNR': 25, 'SSIM': .9, 'LPIPS': .1}}))
                    mesh_path(output, 'radegs').write_text('mesh')
                    return 0
                process.wait.side_effect = finish
                with patch.object(runner, 'ROOT', root), patch.object(runner, 'has_colmap_model', return_value=True), \
                     patch.object(runner, 'scene_identity', return_value={'dataset': 'fixture'}), \
                     patch.object(runner, 'ThesisLogger'), \
                     patch.object(runner.subprocess, 'run', side_effect=subprocess_run), \
                     patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                    runner.run(args)
                    train = popen.call_args_list[0].args[0]
                    self.assertIn('--enable_allocator_gc', train)
                    self.assertEqual(train[train.index('--resolution') + 1], '1920')
                    self.assertNotIn('-r', train)
                    self.assertTrue(any('--guard' in command for command in calls))
                    receipt = json.loads((output / 'togoglab_run.json').read_text())
                    self.assertEqual(receipt['identity']['resolution'], 1920)
                    self.assertTrue((output / 'radegs_effective_config.json').is_file())
                    plan['config']['training']['extra_args'].append('--quiet')
                    popen.reset_mock()
                    with self.assertRaisesRegex(ValueError, 'Konfigurasi/dataset berubah'):
                        runner.run(args)
                    popen.assert_not_called()


class QueueProfileTests(unittest.TestCase):
    def test_new_jobs_select_yaml_and_legacy_jobs_remain_legacy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene = root / 'datasets/Dewa/model_01'
            (scene / 'images').mkdir(parents=True)
            (scene / 'images/a.jpg').write_text('image')
            (scene / 'sparse/0').mkdir(parents=True)
            for name in ('cameras', 'images', 'points3D'):
                (scene / f'sparse/0/{name}.bin').write_text('colmap')
            (root / 'external/RaDe-GS').mkdir(parents=True)
            (root / 'external/RaDe-GS/train.py').write_text('')
            for action in ('radegs', 'full_pipeline'):
                jobs, _ = prepare_jobs(dict(action=action, class_name='Dewa', model_id='model_01',
                                            external_python=sys.executable), root)
                job = jobs[0]
                self.assertEqual(job['radegs_profile'], 'yaml')
                command = build_command(job, root)
                self.assertEqual(command[command.index('--radegs_profile') + 1], 'yaml')
                del job['radegs_profile']
                command = build_command(job, root)
                self.assertEqual(command[command.index('--radegs_profile') + 1], 'legacy')


if __name__ == '__main__':
    unittest.main()
