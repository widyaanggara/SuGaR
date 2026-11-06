"""CPU-only regressions for dataset routing, FIFO isolation, and the HTTP API."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch

from pipeline_tools.dataset_registry import CATALOG, file_signature, has_colmap_model, has_sugar_result, media_files, scene_pairs
from pipeline_tools.init_dataset_folders import initialize_workspace
from sugar_utils.artifact_paths import refined_assets
from web_dashboard.job_queue import JobQueue
from web_dashboard.server import DashboardHandler, build_command, get_project_status, prepare_jobs
from pipeline_tools.run_settings import colmap_signature


def write(path, text='data'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('Timed out waiting for queue condition')


def spec(model='model_01', action='full_pipeline', method='SuGaR'):
    return dict(action=action, class_name='Dewa', model_id=model, evaluation_method=method, gpu=0)


class WorkspaceTests(unittest.TestCase):
    def test_existing_colmap_bulk_start_filters_incomplete_models_and_freezes_start_point(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'datasets/Dewa/model_01'
            write(scene / 'images/imported.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            write(root / 'datasets/Dewa/model_02/input/a.jpg')
            jobs, skipped = prepare_jobs(dict(action='full_pipeline', pipeline_start='existing_colmap',
                colmap_max_image_size=2000, colmap_device='auto', resume=False), root)
            self.assertEqual(len(jobs), 1)
            self.assertTrue(any('Dewa/model_02' in reason and 'COLMAP' in reason for reason in skipped))
            self.assertEqual(jobs[0]['pipeline_start'], 'existing_colmap')
            command = build_command(jobs[0], root)
            self.assertIn('--skip_colmap', command)
            self.assertIn('--no_resume', command)
            self.assertTrue(command[2].endswith('batch_run_all.py'))
            automatic, _ = prepare_jobs(dict(action='full_pipeline'), root)
            self.assertEqual(len(automatic), 2)
            self.assertNotIn('--skip_colmap', build_command(automatic[0], root))
            for invalid in (dict(pipeline_start=True), dict(pipeline_start='wrong'),
                            dict(action='3dgs', class_name='Dewa', model_id='model_01', pipeline_start='existing_colmap')):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    prepare_jobs(invalid, root)

    def test_18_slots_idempotent_and_markers_are_not_photos(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(initialize_workspace(root), 18)
            photo = root / 'datasets/Dewa/model_01/input/keep.JPG'
            write(photo, 'original')
            initialize_workspace(root)
            self.assertEqual(photo.read_text(), 'original')
            self.assertEqual(len(scene_pairs()), 18)
            self.assertEqual(CATALOG['TanksAndTemples'], ('tanks_and_temples_1', 'tanks_and_temples_2'))
            for cls, mdl in scene_pairs():
                self.assertTrue((root / 'datasets' / cls / mdl / 'input').is_dir())
                self.assertTrue((root / 'outputs/sugar' / cls / mdl / 'textured_mesh').is_dir())
            status = get_project_status(root)
            self.assertEqual(status['Dewa']['model_01']['images_count'], 1)
            self.assertEqual(status['Dewa']['model_02']['images_count'], 0)
            self.assertFalse(status['Dewa']['model_02']['3dgs_30k'])
            jobs, skipped = prepare_jobs({'action': 'full_pipeline'}, root)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(len(skipped), 17)

    def test_validate_class_model_and_numeric_options(self):
        for data in ({'class_name': '../Dewa'},
                     {'class_name': 'Dewa', 'model_id': 'tanks_and_temples_1'},
                     {'action': 'evaluate', 'class_name': 'Dewa', 'model_id': 'model_01', 'scale_factor': 'nan'},
                     {'gpu': -1}, {'action': 'shell'}, {'vertices': 3}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                prepare_jobs(data)

    def test_colmap_must_have_all_files_and_undistorted_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            self.assertFalse(has_colmap_model(scene))
            write(scene / 'images/a.jpg')
            self.assertTrue(has_colmap_model(scene))
            (scene / 'sparse/0/images.bin').write_bytes(b'')
            self.assertFalse(has_colmap_model(scene))

    def test_export_resolves_folder_name_and_explicit_mesh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / 'outputs/sugar/Dewa/model_01/refined/sugarfine_test_normalconsistency01_gaussperface1/2000.pt'
            mesh = root / 'outputs/sugar/Dewa/model_01/mesh/sugarmesh_test.ply'
            write(checkpoint)
            write(mesh)
            name, resolved = refined_assets(checkpoint, root / 'datasets/Dewa/model_01', mesh)
            self.assertEqual(name, checkpoint.parent.name)
            self.assertEqual(resolved, mesh.resolve())
            self.assertEqual(refined_assets(checkpoint, root / 'datasets/Dewa/model_01')[1], mesh.resolve())
            with self.assertRaises(FileNotFoundError):
                refined_assets(checkpoint, root, root / 'missing.ply')

    def test_commands_route_benchmarks_and_do_not_double_mask_gpu(self):
        job = dict(action='full_pipeline', class_name='TanksAndTemples', model_id='tanks_and_temples_1', gpu=2)
        command = build_command(job)
        self.assertIn('tanks_and_temples_1', command)
        self.assertEqual(command[command.index('--gpu') + 1], '2')
        self.assertNotIn('CUDA_VISIBLE_DEVICES', ' '.join(command))


class QueueTests(unittest.TestCase):
    def test_restart_retry_stays_paused_and_recovery_is_first_after_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'outputs/queue/state.json', json.dumps({'paused': False, 'jobs': [
                dict(spec('model_01'), id='interrupted', status='RUNNING', created_at=1, started_at=1),
                dict(spec('model_02'), id='waiting', status='QUEUED', created_at=2, started_at=None)]}))
            queue = JobQueue(root, lambda _: self.fail('Must remain paused'), start_worker=False)
            try:
                self.assertTrue(queue.paused)
                self.assertEqual(queue.jobs[0]['status'], 'INTERRUPTED')
                result = queue.retry('interrupted')
                snapshot = queue.snapshot()
                self.assertTrue(snapshot['paused'])
                self.assertFalse(snapshot['is_running'])
                positions = {j['id']: j['queue_position'] for j in snapshot['jobs']}
                self.assertEqual(positions[result['job_id']], 1)
                self.assertEqual(positions['waiting'], 2)
            finally:
                queue.close()
            events = root / 'order.txt'
            def builder(job):
                code = ('from pathlib import Path; '
                        f'Path({str(events)!r}).open("a").write({job["model_id"]!r}+"\\n")')
                return [sys.executable, '-c', code]
            # Priority survives another server restart before the user resumes.
            reopened = JobQueue(root, builder)
            try:
                self.assertTrue(reopened.paused)
                self.assertFalse(events.exists())
                reopened.set_paused(False)
                wait_until(lambda: reopened.snapshot()['counts'].get('COMPLETED') == 2)
                self.assertEqual(events.read_text().splitlines(), ['model_01', 'model_02'])
            finally:
                reopened.close()

    def test_recovery_runs_after_active_job_and_before_fresh_pending_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events, release = root / 'events', root / 'release'
            def builder(job):
                code = ('from pathlib import Path; import time; '
                        f'p=Path({str(events)!r}); '
                        f'p.open("a").write({job["model_id"]!r}+"\\n"); ')
                if job['model_id'] == 'active':
                    code += f'release=Path({str(release)!r}); exec("while not release.exists():\\n time.sleep(.01)"); '
                return [sys.executable, '-c', code]
            queue = JobQueue(root, builder)
            try:
                queue.enqueue([spec('active'), spec('fresh')])
                wait_until(lambda: events.exists())
                active_id = queue.active_id
                with queue.condition:
                    queue.jobs.append(dict(spec('recovery'), id='old', status='INTERRUPTED',
                                           created_at=0, started_at=0, finished_at=0))
                queue.retry('old')
                self.assertEqual(queue.active_id, active_id, 'Recovery must not preempt active work')
                release.write_text('go')
                wait_until(lambda: queue.snapshot()['counts'].get('COMPLETED') == 3)
                self.assertEqual(events.read_text().splitlines(), ['active', 'recovery', 'fresh'])
            finally:
                queue.close()

    def test_real_subprocesses_are_fifo_and_failures_continue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events = root / 'events.jsonl'
            def builder(job):
                code = ("import pathlib,time,json,sys; "
                    f"p=pathlib.Path({str(events)!r}); "
                    f"p.open('a').write(json.dumps(['start',{job['model_id']!r},time.time()])+'\\n'); "
                    "time.sleep(0.12); "
                    f"p.open('a').write(json.dumps(['end',{job['model_id']!r},time.time()])+'\\n'); "
                    f"sys.exit({2 if job['model_id'] == 'bad' else 0})")
                return [sys.executable, '-u', '-c', code]
            queue = JobQueue(root, builder)
            try:
                queue.enqueue([spec('first'), spec('bad'), spec('last')])
                wait_until(lambda: sum(queue.snapshot()['counts'].get(state, 0) for state in ('COMPLETED', 'FAILED')) == 3)
                rows = [json.loads(line) for line in events.read_text().splitlines()]
                self.assertEqual([(r[0], r[1]) for r in rows], [('start', 'first'), ('end', 'first'),
                    ('start', 'bad'), ('end', 'bad'), ('start', 'last'), ('end', 'last')])
                self.assertEqual(queue.snapshot()['counts']['FAILED'], 1)
                self.assertEqual(queue.snapshot()['counts']['COMPLETED'], 2)
                state = json.loads(queue.state_path.read_text())
                self.assertEqual(state['jobs'][1]['returncode'], 2)
                self.assertTrue(queue.job_logs(state['jobs'][0]['id'])['logs'])
            finally:
                queue.close()

    def test_pause_cancel_dedup_and_separate_visual_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = JobQueue(temporary, lambda _: [sys.executable, '-c', 'pass'])
            try:
                queue.set_paused(True)
                added, _ = queue.enqueue([spec(), spec(), spec(action='evaluate_visual', method='3DGS'), spec(action='evaluate_visual')])
                self.assertEqual(len(added), 3)
                self.assertFalse(queue.snapshot()['is_running'])
                self.assertEqual(queue.cancel_pending(added[0]), 1)
                queue.set_paused(False)
                wait_until(lambda: queue.snapshot()['counts'].get('COMPLETED') == 2)
                self.assertEqual(queue.snapshot()['counts']['CANCELLED'], 1)
            finally:
                queue.close()

    def test_restart_marks_active_interrupted_and_keeps_pending_paused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'outputs/queue/state.json', json.dumps({'paused': False, 'jobs': [
                dict(spec('first'), id='first', status='RUNNING', created_at=0, started_at=0),
                dict(spec('second'), id='second', status='QUEUED', created_at=0, started_at=None)]}))
            queue = JobQueue(root, lambda _: [sys.executable, '-c', 'pass'])
            try:
                state = queue.snapshot()
                self.assertTrue(state['paused'])
                self.assertEqual(state['counts']['INTERRUPTED'], 1)
                self.assertEqual(state['counts']['QUEUED'], 1)
            finally:
                queue.close()

    def test_stop_kills_child_before_resuming_next_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heartbeat = root / 'heartbeat'
            child_code = ("import pathlib,time; p=pathlib.Path(" + repr(str(heartbeat)) + "); "
                          "exec('while True:\\n p.write_text(str(time.time()))\\n time.sleep(0.05)')")
            def builder(job):
                if job['model_id'] == 'long':
                    code = f'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",{child_code!r}]); time.sleep(60)'
                else:
                    code = f'from pathlib import Path; Path({str(root / "next")!r}).write_text("done")'
                return [sys.executable, '-u', '-c', code]
            queue = JobQueue(root, builder)
            try:
                queue.enqueue([spec('long'), spec('next')])
                wait_until(lambda: heartbeat.exists())
                queue.stop()
                self.assertTrue(queue.snapshot()['paused'])
                self.assertFalse(queue.snapshot()['is_running'])
                self.assertFalse((root / 'next').exists())
                before = heartbeat.read_text()
                time.sleep(0.2)
                self.assertEqual(heartbeat.read_text(), before)
                queue.set_paused(False)
                wait_until(lambda: queue.snapshot()['counts'].get('COMPLETED') == 1)
                self.assertTrue((root / 'next').exists())
                self.assertEqual(queue.snapshot()['counts']['CANCELLED'], 1)
            finally:
                queue.close()


class RunnerTests(unittest.TestCase):
    def test_automatic_pipeline_reuses_imported_colmap_despite_missing_or_foreign_receipt(self):
        from pipeline_tools import batch_run_all as batch
        for receipt in (None, '{"max_image_size": 4000}', 'invalid receipt'):
            with self.subTest(receipt=receipt), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                scene = root / 'datasets/Dewa/model_01'
                write(scene / 'images/imported.jpg')
                for name in ('cameras', 'images', 'points3D'):
                    write(scene / 'sparse/0' / (name + '.bin'))
                if receipt is not None:
                    write(scene / 'colmap_settings.json', receipt)
                before = {path: file_signature(path) for path in scene.rglob('*') if path.is_file()}
                with patch.object(batch, 'run_stage') as stage:
                    result = batch.process_single_statue(scene, 'Dewa', 'model_01', root=root,
                        colmap_max_image_size=2000, colmap_device='auto')
                self.assertEqual(result, 'COMPLETED')
                self.assertEqual([call.args[0] for call in stage.call_args_list],
                    ['03_train_3dgs.py', '04_train_sugar.py', '06_evaluate_rendering.py', '06_evaluate_rendering.py'])
                self.assertTrue(all('--no_resume' not in call.args[1] for call in stage.call_args_list))
                self.assertEqual(before, {path: file_signature(path) for path in before})

    def test_automatic_pipeline_keeps_completed_3dgs_and_passes_partial_sugar_to_recovery(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'datasets/Dewa/model_01'
            write(scene / 'images/imported.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            baseline = root / 'outputs/3dgs/Dewa/model_01'
            write(baseline / 'cameras.json', '[]')
            for iteration in (7000, 30000):
                write(baseline / f'point_cloud/iteration_{iteration}/point_cloud.ply')
            write(baseline / 'training_settings.json', json.dumps(dict(image_downscale=1,
                image_max_size=1600, colmap_signature=colmap_signature(scene))))
            partial = root / 'outputs/sugar/Dewa/model_01/recovery/snapshot.pt'
            write(partial, 'saved state must remain untouched by the batch runner')
            before = file_signature(partial)
            with patch.object(batch, 'run_stage') as stage:
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root, colmap_max_image_size=2000)
            self.assertEqual([call.args[0] for call in stage.call_args_list],
                ['04_train_sugar.py', '06_evaluate_rendering.py', '06_evaluate_rendering.py'])
            self.assertNotIn('--no_resume', stage.call_args_list[0].args[1])
            self.assertEqual(file_signature(partial), before)

    def test_disabling_resume_explicitly_rebuilds_existing_colmap(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary)
            write(scene / 'images/imported.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            with patch.object(batch, 'run_stage', side_effect=subprocess.CalledProcessError(2, 'colmap')) as stage:
                with self.assertRaises(subprocess.CalledProcessError):
                    batch.process_single_statue(scene, 'Dewa', 'model_01', resume=False)
            self.assertEqual(stage.call_args.args[0], '02_run_colmap.py')

    def test_explicit_imported_colmap_runs_all_remaining_stages_without_colmap_or_input_photos(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'datasets/Dewa/model_01'
            write(scene / 'images/imported.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / 'sparse/0' / (name + '.bin'))
            # A receipt copied from elsewhere may have unknown/mismatched metadata.
            write(scene / 'colmap_settings.json', '{"max_image_size": 4000}')
            before = {path: file_signature(path) for path in scene.rglob('*') if path.is_file()}
            with patch.object(batch, 'run_stage') as stage:
                result = batch.process_single_statue(scene, 'Dewa', 'model_01', root=root,
                    skip_colmap=True, resume=False, colmap_max_image_size=2000, colmap_device='auto')
            self.assertEqual(result, 'COMPLETED')
            self.assertEqual([call.args[0] for call in stage.call_args_list],
                ['03_train_3dgs.py', '04_train_sugar.py', '06_evaluate_rendering.py', '06_evaluate_rendering.py'])
            methods = [call.args[1][call.args[1].index('--method') + 1] for call in stage.call_args_list[-2:]]
            self.assertEqual(methods, ['3DGS', 'SuGaR'])
            self.assertEqual(before, {path: file_signature(path) for path in before})

    def test_missing_imported_colmap_fails_before_any_stage_even_if_raw_video_exists(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'raw_video/video.mp4')
            write(root / 'images/imported.jpg')
            write(root / 'sparse/0/cameras.bin')
            with patch.object(batch, 'run_stage') as stage, self.assertRaisesRegex(RuntimeError, 'COLMAP tidak dijalankan ulang'):
                batch.process_single_statue(root, 'Dewa', 'model_01', skip_colmap=True)
            stage.assert_not_called()

    def test_stages_and_both_evaluations_then_resume(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = root / 'datasets/Dewa/model_01'
            write(scene / 'input/a.jpg')
            baseline = root / 'outputs/3dgs/Dewa/model_01'
            sugar = root / 'outputs/sugar/Dewa/model_01'
            calls = []
            def fake_stage(script, args, root_arg):
                calls.append(script)
                if script == '02_run_colmap.py':
                    write(scene / 'images/a.jpg')
                    for name in ('cameras', 'images', 'points3D'):
                        write(scene / 'sparse/0' / (name + '.bin'))
                elif script == '03_train_3dgs.py':
                    write(baseline / 'cameras.json', '[]')
                    for iteration in (7000, 30000):
                        write(baseline / f'point_cloud/iteration_{iteration}/point_cloud.ply')
                    write(baseline / 'training_settings.json', json.dumps(dict(image_downscale=1,
                        image_max_size=1600, colmap_signature=colmap_signature(scene))))
                elif script == '04_train_sugar.py':
                    for name in ('mesh/test.ply', 'refined/test/2000.pt', 'textured_mesh/test.obj'):
                        write(sugar / name)
                    manifest = dict(coarse_mesh='mesh/test.ply', refined_checkpoint='refined/test/2000.pt',
                        textured_mesh='textured_mesh/test.obj', target_triangles=250000,
                        refinement_iterations=2000, regularization='dn_consistency', image_downscale=1, image_max_size=1600,
                        gaussians_per_triangle=1)
                    write(sugar / 'pipeline_result.json', json.dumps(manifest))
                elif script == '06_evaluate_rendering.py':
                    method = args[args.index('--method') + 1]
                    output = baseline if method == '3DGS' else sugar
                    checkpoint = 'point_cloud/iteration_30000/point_cloud.ply' if method == '3DGS' else 'refined/test/2000.pt'
                    write(output / 'evaluation/metrics.json', json.dumps(dict(psnr=20, ssim=.8, lpips=.2,
                        image_downscale=1, image_max_size=1600,
                        checkpoint=checkpoint, checkpoint_signature=file_signature(output / checkpoint), per_view=[{}])))
            with patch.object(batch, 'run_stage', side_effect=fake_stage):
                self.assertEqual(batch.process_single_statue(scene, 'Dewa', 'model_01', root=root), 'COMPLETED')
                self.assertEqual(calls, ['02_run_colmap.py', '03_train_3dgs.py', '04_train_sugar.py',
                                         '06_evaluate_rendering.py', '06_evaluate_rendering.py'])
                calls.clear()
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root)
                self.assertEqual(calls, [])
                # Changing refinement invalidates the completed SuGaR stage.
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root, refinement_time='medium')
                self.assertEqual(calls, ['04_train_sugar.py', '06_evaluate_rendering.py'])
                calls.clear()
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root, gaussians_per_triangle=6)
                self.assertEqual(calls, ['04_train_sugar.py', '06_evaluate_rendering.py'])
                calls.clear()
                batch.process_single_statue(scene, 'Dewa', 'model_01', root=root, regularization='density')
                self.assertEqual(calls, ['04_train_sugar.py', '06_evaluate_rendering.py'])

    def test_stage_failure_propagates_and_main_reports_failure(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'input/a.jpg')
            with patch.object(batch, 'run_stage', side_effect=subprocess.CalledProcessError(2, 'colmap')):
                with self.assertRaises(subprocess.CalledProcessError):
                    batch.process_single_statue(root, 'Dewa', 'model_01')
        with patch.object(sys, 'argv', ['batch_run_all.py', '-c', 'Dewa']), \
             patch.object(batch, 'process_single_statue', side_effect=[RuntimeError('failed'), 'COMPLETED']):
            self.assertEqual(batch.main(), 1)

    def test_full_pipeline_forwards_custom_colmap_settings_to_the_actual_stage(self):
        from pipeline_tools import batch_run_all as batch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root / 'input/a.jpg')
            with patch.object(batch, 'run_stage', side_effect=subprocess.CalledProcessError(2, 'colmap')) as stage:
                with self.assertRaises(subprocess.CalledProcessError):
                    batch.process_single_statue(root, 'Dewa', 'model_01', gpu=2,
                        colmap_device='auto', colmap_matcher='sequential',
                        colmap_max_image_size=2000, colmap_max_features=4096, colmap_sequential_overlap=15)
                script, args, _ = stage.call_args.args
                self.assertEqual(script, '02_run_colmap.py')
                for flag, expected in (('--gpu', 2), ('--colmap_device', 'auto'),
                    ('--colmap_matcher', 'sequential'), ('--colmap_max_features', '4096'),
                    ('--colmap_sequential_overlap', '15'), ('--colmap_max_image_size', 2000)):
                    self.assertEqual(args[args.index(flag) + 1], expected)

    def test_duration_does_not_sum_nested_timers(self):
        from pipeline_tools.logger_utils import ThesisLogger
        with tempfile.TemporaryDirectory() as temporary:
            with patch('pipeline_tools.logger_utils.time.monotonic', side_effect=[100., 130.]):
                logger = ThesisLogger('Dewa', 'model_01', base_dir=temporary)
                try:
                    logger.data['timings_seconds'] = {'total': 30, 'feature': 10, 'mapping': 20}
                    logger.finalize()
                    self.assertEqual(logger.data['total_duration_seconds'], 30)
                    self.assertEqual(logger.data['vram_measurement'], 'unavailable')
                finally:
                    for handler in logger.logger.handlers:
                        handler.close()


class HttpTests(unittest.TestCase):
    def test_api_all_18_and_csv_download_and_bad_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_workspace(root)
            for cls, mdl in scene_pairs():
                write(root / 'datasets' / cls / mdl / 'input/a.JPG')
            write(root / 'outputs/logs/evaluation_reports/summary_comparison.csv', 'Kelas,Metode\nDewa,SuGaR\nDewa,SuGaR_scaffold_anchor\n')
            queue = JobQueue(root, lambda _: [sys.executable, '-c', 'pass'])
            queue.set_paused(True)
            server = HTTPServer(('127.0.0.1', 0), DashboardHandler)
            server.job_queue, server.project_root = queue, root
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            def request(path, body=None):
                req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                             headers={'Content-Type': 'application/json'})
                return urllib.request.urlopen(req, timeout=10)
            try:
                with request('/api/catalog') as response:
                    self.assertEqual(json.load(response)['total_models'], 18)
                with request('/api/model-name', dict(class_name='Dewa', model_id='model_01', display_name='Ganesha')) as response:
                    self.assertEqual(json.load(response)['display_name'], 'Ganesha')
                with request('/api/status') as response:
                    model = json.load(response)['Dewa']['model_01']
                    self.assertEqual(model['display_name'], 'Ganesha')
                    self.assertEqual(model['input_path'], 'datasets/Dewa/model_01/input/')
                self.assertEqual(queue.snapshot()['counts'].get('QUEUED', 0), 0)
                for body in ({}, dict(class_name='Dewa', model_id='../model_01', display_name='Name'),
                             dict(class_name='Dewa', model_id='model_01', display_name=['Name'])):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        request('/api/model-name', body)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                with request('/api/presets') as response:
                    self.assertEqual(json.load(response)['presets']['official_low']['values']['gaussians_per_triangle'], 6)
                with patch('web_dashboard.server.inspect_colmap', return_value=dict(available=True, cuda=False)):
                    with request('/api/colmap') as response:
                        self.assertEqual(json.load(response), dict(available=True, cuda=False))
                with request('/api/run', {'action': 'full_pipeline'}) as response:
                    self.assertEqual(response.status, 202)
                    job_ids = json.load(response)['added']
                    self.assertEqual(len(job_ids), 18)
                with request('/api/jobs/' + job_ids[0] + '/logs') as response:
                    self.assertEqual(json.load(response)['job']['status'], 'QUEUED')
                with request('/api/queue') as response:
                    self.assertEqual(json.load(response)['counts']['QUEUED'], 18)
                with request('/api/run', {'action': 'full_pipeline'}) as response:
                    self.assertEqual(len(json.load(response)['duplicates']), 18)
                queue.jobs[0]['status'] = 'INTERRUPTED'
                with request('/api/queue/retry', {'job_id': job_ids[0], 'gpu': 999}) as response:
                    recovered = json.load(response)
                    self.assertTrue(recovered['promoted'])
                    recovery_job = next(job for job in queue.jobs if job['id'] == recovered['job_id'])
                    self.assertEqual(recovery_job['gpu'], 0, 'Retry must use server-side original settings')
                    self.assertTrue(recovery_job['resume'])
                with request('/api/queue/retry', {'job_id': job_ids[0]}) as response:
                    self.assertEqual(json.load(response)['job_id'], recovered['job_id'])
                self.assertTrue(queue.paused)
                for body in ({'job_id': 'missing'}, {'job_id': []}, {}):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        request('/api/queue/retry', body)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                with request('/api/metrics.csv') as response:
                    self.assertIn('attachment', response.headers['Content-Disposition'])
                    csv_data = response.read()
                    self.assertIn(b'Dewa,SuGaR', csv_data)
                    self.assertIn(b'Nama Model', csv_data)
                with request('/api/metrics.csv?experiment=baseline') as response:
                    self.assertNotIn(b'scaffold_anchor', response.read())
                with request('/api/metrics.csv?experiment=scaffold_anchor&method=SuGaR') as response:
                    csv_data = response.read()
                    self.assertIn(b'SuGaR_scaffold_anchor', csv_data)
                    self.assertNotIn(b'Dewa,SuGaR,', csv_data)
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request('/api/metrics.csv?experiment=invalid')
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
                for body in ({'action': 'execute'}, [], {'class_name': '..'}):
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        request('/api/run', body)
                    self.assertEqual(error.exception.code, 400)
                    error.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request('/static/outputs/../../secret.txt')
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                queue.close()


if __name__ == '__main__':
    unittest.main()
