"""CPU regression tests for the anchor-initialization experiment."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from pipeline_tools import batch_run_all as batch
from pipeline_tools.scaffold_init import export_anchors, prepare
from pipeline_tools.scaffold_train import training_module
from pipeline_tools.run_settings import training_matches, colmap_signature
from pipeline_tools.dataset_registry import file_signature
from web_dashboard.server import prepare_jobs, build_command
from web_dashboard.job_queue import JobQueue


def write(path, content='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def scene_at(root):
    scene = root / 'datasets/Dewa/model_01'
    write(scene / 'images/a.jpg')
    for name in ('cameras', 'images'):
        write(scene / f'sparse/0/{name}.txt')
    write(scene / 'sparse/0/points3D.txt', '1 0 0 0 255 0 0 0\n2 10 0 0 0 255 0 0\n')
    return scene


def anchors(path, rows='0.1 0 0 1\n9.9 0 0 2\n0.1 0 0 1\n', count=3):
    write(path, f'ply\nformat ascii 1.0\nelement vertex {count}\nproperty float x\nproperty float y\nproperty float z\nproperty float f_anchor_feat_0\nend_header\n' + rows)


class ScaffoldTests(unittest.TestCase):
    def test_anchor_conversion_preserves_coordinates_colors_and_source(self):
        from plyfile import PlyData
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = scene_at(root)
            source = root / 'anchors.ply'
            anchors(source)
            before = (scene / 'sparse/0/points3D.txt').read_bytes()
            receipt = export_anchors(source, scene, root / 'initial.ply')
            result = PlyData.read(root / 'initial.ply', mmap=False)['vertex'].data
            self.assertEqual(receipt['point_count'], 2)
            self.assertEqual(result['red'].tolist(), [255, 0])
            self.assertEqual(result['green'].tolist(), [0, 255])
            self.assertAlmostEqual(float(result['x'][1]), 9.9, places=5)
            self.assertEqual(before, (scene / 'sparse/0/points3D.txt').read_bytes())
            anchors(source, 'nan 0 0 1\n1 0 0 2\n', 2)
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                export_anchors(source, scene, root / 'invalid.ply')

    def test_stage_reuse_and_changed_inputs_retrain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = scene_at(root)
            repo = root / 'Scaffold-GS'
            for name in ('train.py', 'scene/gaussian_model.py', 'arguments/__init__.py'):
                write(repo / name)
            def fake_run(command, **kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
                self.assertIn('--eval', command)
                output = Path(command[command.index('-m') + 1])
                anchors(output / 'point_cloud/iteration_30000/point_cloud.ply')
            with patch('pipeline_tools.scaffold_init.subprocess.run', side_effect=fake_run) as run, \
                 patch('pipeline_tools.scaffold_init.gpu_lock'):
                first = prepare(scene, root / 'output', repo)
                self.assertEqual(prepare(scene, root / 'output', repo), first)
                self.assertEqual(run.call_count, 1)
                write(scene / 'images/a.jpg', 'changed photo')
                self.assertNotEqual(prepare(scene, root / 'output', repo), first)
                self.assertEqual(run.call_count, 2)

    def test_adapter_does_not_run_upstream_gpu_detection_or_main(self):
        source = '''import os
os.environ['CUDA_VISIBLE_DEVICES'] = 'wrong'
raise RuntimeError('GPU detection should not run')
import torch
def training(): pass
if __name__ == '__main__': raise RuntimeError('main should not run')
'''
        with patch.dict(sys.modules, {'torch': object()}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
            namespace = {'__name__': 'scaffold_training'}
            exec(compile(training_module(source), '<test>', 'exec'), namespace)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')
            self.assertTrue(callable(namespace['training']))

    def test_batch_routes_downstream_to_experiment_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = scene_at(root)
            initial = root / 'initial.ply'
            write(initial)
            with patch('pipeline_tools.scaffold_init.prepare', return_value=str(initial)), \
                 patch.object(batch, 'run_stage') as run:
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root,
                    skip_colmap=True, experiment='scaffold_anchor', scaffold_repo='repo')
                self.assertEqual([c.args[0] for c in run.call_args_list],
                    ['03_train_3dgs.py', '04_train_sugar.py', '06_evaluate_rendering.py', '06_evaluate_rendering.py'])
                arguments = run.call_args_list[0].args[1]
                self.assertEqual(arguments[arguments.index('--initial_ply') + 1], str(initial))
                self.assertIn('ablations', str(arguments[arguments.index('-o') + 1]))
                for call in run.call_args_list:
                    self.assertIn('--experiment', call.args[1])
            with patch('pipeline_tools.scaffold_init.prepare', side_effect=RuntimeError('Scaffold failed')), \
                 patch.object(batch, 'run_stage') as run:
                with self.assertRaisesRegex(RuntimeError, 'Scaffold failed'):
                    batch.process_single_statue(scene, 'Dewa', 'model_01', root=root,
                        skip_colmap=True, experiment='scaffold_anchor', scaffold_repo='repo')
                run.assert_not_called()

    def test_initialization_invalidates_completed_3dgs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = scene_at(root)
            initial = root / 'initial.ply'
            write(initial)
            settings = dict(image_downscale=1, image_max_size=1600, colmap_signature=colmap_signature(scene),
                initialization={'path': str(initial.resolve()), 'signature': file_signature(initial)})
            write(root / 'model/training_settings.json', json.dumps(settings))
            self.assertTrue(training_matches(root / 'model', scene, 1, 1600, initial_ply=initial))
            self.assertFalse(training_matches(root / 'model', scene, 1, 1600))
            self.assertTrue(training_matches(root / 'model', scene, 1, 1600, use_recorded_initialization=True))
            write(initial, 'modified initialization')
            self.assertFalse(training_matches(root / 'model', scene, 1, 1600, initial_ply=initial))
            self.assertFalse(training_matches(root / 'model', scene, 1, 1600, use_recorded_initialization=True))

    def test_dashboard_queue_keeps_baseline_and_ablation_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_at(root)
            repo = root / 'Scaffold-GS'
            write(repo / 'train.py')
            baseline, _ = prepare_jobs(dict(action='full_pipeline', class_name='Dewa', model_id='model_01'), root)
            ablation, _ = prepare_jobs(dict(action='full_pipeline', class_name='Dewa', model_id='model_01',
                experiment='scaffold_anchor', scaffold_repo=str(repo), pipeline_start='existing_colmap'), root)
            command = build_command(ablation[0], root)
            self.assertIn('--skip_colmap', command)
            self.assertIn('--scaffold_repo', command)
            queue = JobQueue(root, lambda job: [sys.executable, '-c', 'pass'])
            queue.set_paused(True)
            try:
                added, duplicate = queue.enqueue(baseline + ablation)
                self.assertEqual(len(added), 2)
                self.assertEqual(duplicate, [])
                self.assertEqual(len(queue.enqueue(ablation)[1]), 1)
            finally:
                queue.close()


if __name__ == '__main__':
    unittest.main()
