"""Scaffold controls must reach training and invalidate cached anchors."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline_tools.scaffold_settings import scaffold_settings, scaffold_arguments
from pipeline_tools.scaffold_init import prepare
from web_dashboard.server import prepare_jobs, build_command


def write(path, text='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class ScaffoldSettingsTests(unittest.TestCase):
    def test_reject_invalid_values_and_growth_schedule(self):
        for values in ({'scaffold_voxel_size': float('nan')}, {'scaffold_feat_dim': True},
                       {'scaffold_n_offsets': 1.5}, {'scaffold_lambda_dssim': 2},
                       {'scaffold_update_interval': 0}, {'scaffold_update_from': 20000},
                       {'scaffold_update_depth': 4}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                scaffold_settings(values)

    def test_api_to_batch_command_and_upstream_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'datasets/Dewa/model_01/input/a.jpg')
            repo = root / 'scaffold'
            write(repo / 'train.py')
            values = dict(scaffold_voxel_size=0.002, scaffold_n_offsets=6)
            with patch('pipeline_tools.batch_run_all.pipeline_complete', return_value=True):
                jobs, _ = prepare_jobs(dict(action='full_pipeline', experiment='scaffold_anchor',
                    scaffold_repo=str(repo), class_name='Dewa', model_id='model_01', **values), root)
            self.assertEqual(len(jobs), 1)
            command = build_command(jobs[0], root)
            for key, value in values.items():
                self.assertEqual(command[command.index('--' + key) + 1], str(value))
            upstream = scaffold_arguments(jobs[0], upstream=True)
            self.assertEqual(upstream[upstream.index('--voxel_size') + 1], '0.002')
            self.assertNotIn('--scaffold_voxel_size', upstream)

    def test_changed_parameter_retrains_and_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene, repo = root / 'scene', root / 'repo'
            write(scene / 'images/a.jpg')
            for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
                write(scene / 'sparse/0' / name)
            for name in ('train.py', 'scene/gaussian_model.py', 'arguments/__init__.py'):
                write(repo / name)
            def run(command, **kwargs):
                output = Path(command[command.index('-m') + 1])
                write(output / 'point_cloud/iteration_30000/point_cloud.ply')
            def export(source, scene, destination):
                write(Path(destination))
                return {}
            with patch('pipeline_tools.scaffold_init.subprocess.run', side_effect=run) as mocked, \
                 patch('pipeline_tools.scaffold_init.export_anchors', side_effect=export), \
                 patch('pipeline_tools.scaffold_init.gpu_lock'):
                first = prepare(scene, root / 'output', repo)
                self.assertEqual(prepare(scene, root / 'output', repo), first)
                self.assertEqual(mocked.call_count, 1)
                second = prepare(scene, root / 'output', repo, settings={'scaffold_voxel_size': 0.002})
                self.assertNotEqual(second, first)
                self.assertEqual(mocked.call_count, 2)
                command = mocked.call_args.args[0]
                self.assertEqual(command[command.index('--voxel_size') + 1], '0.002')
