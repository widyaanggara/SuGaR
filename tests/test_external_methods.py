import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from pipeline_tools import run_external_method as runner
from pipeline_tools.external_methods import checkpoint, mesh_path
from web_dashboard.server import prepare_jobs, build_command, get_project_status, get_metrics_table
from pipeline_tools.geometry_config import geometry_model_path
from pipeline_tools.batch_run_all import process_single_statue


class ExternalMethodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for repo in ('2DGS', 'RaDe-GS'):
            self.write(self.root / 'external' / repo / 'train.py')
        scene = self.root / 'datasets/Penabuh/model_01'
        self.write(scene / 'images/a.jpg')
        for name in ('cameras', 'images', 'points3D'):
            self.write(scene / ('sparse/0/' + name + '.bin'))

    def write(self, path, value='data'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')

    def test_bulk_queue_routes_methods_and_skips_unprepared_slots(self):
        for method in ('2dgs', 'radegs'):
            jobs, skipped = prepare_jobs(dict(action=method, class_name='Penabuh', external_python=sys.executable), self.root)
            self.assertEqual(len(jobs), 1)
            self.assertTrue(skipped)
            command = build_command(jobs[0], self.root)
            self.assertIn('run_external_method.py', command[2])
            self.assertEqual(command[command.index('--method') + 1], method)
            self.assertEqual(command[command.index('--python') + 1], sys.executable)
            self.assertNotIn('--vertices', command)

    def test_auto_resolution_reaches_native_training_from_queue(self):
        for action in ('radegs', 'full_pipeline'):
            jobs, _ = prepare_jobs(dict(action=action, class_name='Penabuh', model_id='model_01',
                                        external_python=sys.executable, external_resolution=-1), self.root)
            command = build_command(jobs[0], self.root)
            flag = '--external_resolution' if action == 'full_pipeline' else '--resolution'
            self.assertEqual(command[command.index(flag) + 1], '-1')
        stages = dict(runner.commands('radegs', sys.executable, self.root, self.root / 'output', -1))
        self.assertEqual(stages['Training'][stages['Training'].index('-r') + 1], '-1')
        self.assertEqual(stages['Ekstraksi_Mesh'][stages['Ekstraksi_Mesh'].index('-r') + 1], '2')

    def test_new_jobs_default_to_auto_but_existing_jobs_keep_resolution(self):
        for action in ('radegs', 'full_pipeline'):
            jobs, _ = prepare_jobs(dict(action=action, class_name='Penabuh', model_id='model_01',
                                        external_python=sys.executable), self.root)
            self.assertEqual(jobs[0]['external_resolution'], -1)
            flag = '--external_resolution' if action == 'full_pipeline' else '--resolution'
            for resolution in (-1, 1, 2):
                job = dict(jobs[0], external_resolution=resolution)
                command = build_command(job, self.root)
                self.assertEqual(command[command.index(flag) + 1], str(resolution))
            legacy = dict(jobs[0])
            del legacy['external_resolution']
            command = build_command(legacy, self.root)
            self.assertEqual(command[command.index(flag) + 1], '1')

    def test_resolution_change_rejects_resume_before_starting_training(self):
        output = self.root / 'outputs/radegs/Penabuh/model_01'
        identity = {'scene': runner.scene_identity(self.root / 'datasets/Penabuh/model_01'),
                    'resolution': 1, 'experiment': 'baseline', 'initial_sha256': None}
        self.write(output / 'togoglab_run.json', json.dumps({'identity': identity}))
        self.write(output / 'chkpnt15000.pth', 'saved training')
        args = argparse.Namespace(statue_class='Penabuh', model_id='model_01', method='radegs',
                                  gpu=0, python=sys.executable, evaluate_only=False, no_resume=False, resolution=-1)
        with patch.object(runner, 'ROOT', self.root), patch.object(runner.subprocess, 'run'), \
             patch.object(runner.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'Konfigurasi/dataset berubah'):
                runner.run(args)
            popen.assert_not_called()
        self.assertEqual((output / 'chkpnt15000.pth').read_text(), 'saved training')

    def test_missing_python_reports_environment_path_before_writing_outputs(self):
        args = argparse.Namespace(statue_class='Penabuh', model_id='model_01', method='radegs',
                                  gpu=0, python='/env/radegs/bin/pytho', evaluate_only=False,
                                  no_resume=False, resolution=-1)
        with patch.object(runner, 'ROOT', self.root), \
             patch.object(runner.subprocess, 'run', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'Python metode tidak ditemukan'):
                runner.run(args)
        self.assertFalse((self.root / 'outputs/radegs/Penabuh/model_01').exists())

    def test_native_artifacts_status_metrics_and_geometry_paths(self):
        for key, label in (('2dgs', '2DGS'), ('radegs', 'RaDe-GS')):
            output = self.root / 'outputs' / key / 'Penabuh/model_01'
            self.write(checkpoint(output))
            self.write(output / 'cfg_args')
            self.write(output / 'results.json', json.dumps({'ours_30000': {'PSNR': 25, 'SSIM': .9, 'LPIPS': .1}}))
            self.write(mesh_path(output, key))
            jobs, _ = prepare_jobs(dict(action='evaluate_visual', class_name='Penabuh', model_id='model_01',
                                        evaluation_method=label), self.root)
            self.assertIn('--evaluate_only', build_command(jobs[0], self.root))
            self.assertEqual(geometry_model_path(self.root, 'Penabuh', 'model_01', label), mesh_path(output, key))
        status = get_project_status(self.root)['Penabuh']['model_01']
        self.assertTrue(status['external_methods']['2dgs']['trained'])
        self.assertTrue(status['external_methods']['radegs']['evaluated'])
        self.assertEqual({r['Metode'] for r in get_metrics_table(self.root)}, {'2DGS', 'RaDe-GS'})

    def test_runner_skips_training_and_rejects_swallowed_metric_failure(self):
        output = self.root / 'outputs/2dgs/Penabuh/model_01'
        self.write(checkpoint(output))
        self.write(output / 'cfg_args')
        self.write(mesh_path(output, '2dgs'))
        args = argparse.Namespace(statue_class='Penabuh', model_id='model_01', method='2dgs',
                                  gpu=0, python=sys.executable, evaluate_only=False, no_resume=False, resolution=2)
        process = MagicMock()
        process.wait.return_value = 0
        process.poll.return_value = 0
        with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'ThesisLogger'), \
             patch.object(runner.subprocess, 'run'), patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, 'PSNR'):
                runner.run(args)
            self.assertEqual([c.args[0][2] for c in popen.call_args_list], ['render.py', 'metrics.py'])

    def test_commands_are_native_and_use_separate_output(self):
        for method in ('2dgs', 'radegs'):
            output = self.root / 'outputs' / method / 'Penabuh/model_01'
            stages = dict(runner.commands(method, sys.executable, self.root / 'datasets/Penabuh/model_01', output, 4))
            self.assertIn('--eval', stages['Training'])
            self.assertEqual(stages['Training'][stages['Training'].index('-m') + 1], str(output))
            self.assertEqual(stages['Evaluasi_Visual'][2], 'metrics.py' if method == '2dgs' else 'metric.py')
            self.assertEqual('Ekstraksi_Mesh' in stages, method == 'radegs')

    def test_full_pipeline_finishes_all_methods_per_statue_in_both_experiments(self):
        for experiment in ('baseline', 'scaffold_anchor'):
            stages = []
            anchor = self.root / 'anchor.ply'
            self.write(anchor)
            with patch('pipeline_tools.batch_run_all.run_stage', side_effect=lambda script, args, root: stages.append((script, args))), \
                 patch('pipeline_tools.scaffold_init.prepare', return_value=str(anchor)):
                status = process_single_statue(self.root / 'datasets/Penabuh/model_01', 'Penabuh', 'model_01',
                    root=self.root, include_external=True, experiment=experiment,
                    scaffold_repo=str(self.root / 'scaffold'), python_2dgs='/env/2dgs/python', python_radegs='/env/radegs/python')
            self.assertEqual(status, 'COMPLETED')
            self.assertEqual([s for s, _ in stages], ['03_train_3dgs.py', '04_train_sugar.py',
                'run_external_method.py', 'run_external_method.py',
                '06_evaluate_rendering.py', '06_evaluate_rendering.py',
                'run_external_method.py', 'run_external_method.py'])
            for index, method in ((2, '2dgs'), (3, 'radegs'), (6, '2dgs'), (7, 'radegs')):
                args = stages[index][1]
                self.assertEqual(args[args.index('--method') + 1], method)
                self.assertEqual(args[args.index('--experiment') + 1], experiment)
                self.assertIn('--train_only' if index < 4 else '--postprocess_only', args)
                self.assertEqual('--initial_ply' in args, experiment == 'scaffold_anchor')

    def test_full_pipeline_failure_does_not_claim_all_methods_completed(self):
        def fail_external(script, args, root):
            if script == 'run_external_method.py':
                raise RuntimeError('2DGS failed')
        with patch('pipeline_tools.batch_run_all.run_stage', side_effect=fail_external):
            with self.assertRaisesRegex(RuntimeError, '2DGS failed'):
                process_single_statue(self.root / 'datasets/Penabuh/model_01', 'Penabuh', 'model_01',
                                      root=self.root, include_external=True)

    def test_radegs_anchor_staging_preserves_colmap_and_reuses_directory(self):
        scene = self.root / 'datasets/Penabuh/model_01'
        self.write(scene / 'sparse/0/points3D.ply', 'original COLMAP')
        anchor = self.root / 'anchor.ply'
        self.write(anchor, 'scaffold anchors')
        output = self.root / 'outputs/ablations/scaffold_anchor/radegs/Penabuh/model_01'
        staged = runner.staged_anchor_scene(scene, output, anchor)
        self.assertEqual((scene / 'sparse/0/points3D.ply').read_text(), 'original COLMAP')
        self.assertEqual((staged / 'sparse/0/points3D.ply').read_text(), 'scaffold anchors')
        self.assertEqual(runner.staged_anchor_scene(scene, output, anchor), staged)

    def test_legacy_checkpoint_compatibility_is_scoped_to_radegs_resume(self):
        for method, resume in (('radegs', True), ('radegs', False), ('2dgs', True)):
            with self.subTest(method=method, resume=resume):
                output = self.root / 'outputs' / method / 'Penabuh/model_01'
                identity = {'scene': runner.scene_identity(self.root / 'datasets/Penabuh/model_01'),
                            'resolution': 1, 'experiment': 'baseline', 'initial_sha256': None}
                self.write(output / 'togoglab_run.json', json.dumps({'identity': identity}))
                saved = output / 'chkpnt15000.pth'
                self.write(saved, 'local training state')
                checkpoint(output).unlink(missing_ok=True)
                args = argparse.Namespace(statue_class='Penabuh', model_id='model_01', method=method,
                    train_only=True, gpu=0, python=sys.executable, evaluate_only=False,
                    no_resume=not resume, resolution=1)
                process = MagicMock()
                def complete():
                    self.write(checkpoint(output))
                    self.write(output / 'cfg_args')
                    return 0
                process.wait.side_effect = complete
                process.poll.return_value = 0
                with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'ThesisLogger'), \
                     patch.dict(runner.os.environ, {}, clear=True), \
                     patch.object(runner.subprocess, 'run') as preflight, \
                     patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                    runner.run(args)
                    command = popen.call_args.args[0]
                    self.assertEqual('--start_checkpoint' in command, resume)
                    if resume:
                        self.assertEqual(command[command.index('--start_checkpoint') + 1], str(saved))
                    child_env = popen.call_args.kwargs['env']
                    self.assertEqual(child_env.get('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'),
                                     '1' if method == 'radegs' and resume else None)
                    self.assertNotIn('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', preflight.call_args.kwargs['env'])
                    self.assertNotIn('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', runner.os.environ)

    def test_dashboard_full_pipeline_snapshots_all_environments(self):
        jobs, _ = prepare_jobs(dict(action='full_pipeline', class_name='Penabuh', model_id='model_01',
            python_2dgs='/env/2dgs/python', python_radegs='/env/radegs/python'), self.root)
        command = build_command(jobs[0], self.root)
        self.assertIn('--include_external', command)
        self.assertEqual(command[command.index('--python_2dgs') + 1], '/env/2dgs/python')
        self.assertEqual(command[command.index('--python_radegs') + 1], '/env/radegs/python')

    def test_ablation_runner_passes_anchor_without_writing_baseline_output(self):
        anchor = self.root / 'anchor.ply'
        self.write(anchor, 'anchor data')
        for method in ('2dgs', 'radegs'):
            output = self.root / 'outputs/ablations/scaffold_anchor' / method / 'Penabuh/model_01'
            args = argparse.Namespace(statue_class='Penabuh', model_id='model_01', method=method,
                experiment='scaffold_anchor', initial_ply=str(anchor), train_only=True,
                gpu=0, python=sys.executable, evaluate_only=False, no_resume=False, resolution=1)
            process = MagicMock()
            def complete():
                self.write(checkpoint(output))
                self.write(output / 'cfg_args')
                return 0
            process.wait.side_effect = complete
            process.poll.return_value = 0
            with patch.object(runner, 'ROOT', self.root), patch.object(runner, 'ThesisLogger'), \
                 patch.object(runner.subprocess, 'run'), patch.object(runner.subprocess, 'Popen', return_value=process) as popen:
                runner.run(args)
                command = popen.call_args.args[0]
                self.assertEqual(command[command.index('-m') + 1], str(output))
                if method == '2dgs':
                    self.assertEqual(command[command.index('--init_ply') + 1], str(anchor))
                else:
                    source = Path(command[command.index('-s') + 1])
                    self.assertEqual((source / 'sparse/0/points3D.ply').read_text(), 'anchor data')
                popen.reset_mock()
                runner.run(args)
                popen.assert_not_called()  # Completed ablation training is reused on retry.
            self.assertFalse((self.root / 'outputs' / method / 'Penabuh/model_01').exists())


if __name__ == '__main__':
    unittest.main()
