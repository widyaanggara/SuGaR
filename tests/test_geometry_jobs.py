"""CPU-only API tests: no COLMAP, reconstruction, Open3D, or training is run."""
import json
from http.server import HTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from pipeline_tools.dataset_registry import file_signature
from pipeline_tools.geometry_config import (geometry_result_current, read_geometry_scales,
    save_geometry_scales, read_geometry_settings, save_geometry_settings)
from pipeline_tools.init_dataset_folders import initialize_workspace
from web_dashboard.job_queue import JobQueue
from web_dashboard.server import DashboardHandler, build_command, get_project_status, prepare_jobs


def write(path, content='fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def artifacts(root, mdl='model_01'):
    write(root / 'datasets/Dewa' / mdl / 'ground_truth/lidar.ply')
    baseline = root / 'outputs/3dgs/Dewa' / mdl
    write(baseline / 'cameras.json', '{}')
    for iteration in (7000, 30000):
        write(baseline / f'point_cloud/iteration_{iteration}/point_cloud.ply')
    output = root / 'outputs/sugar/Dewa' / mdl
    files = dict(textured_mesh='textured_mesh/statue.obj', coarse_mesh='mesh/coarse.ply',
                 refined_checkpoint='refined/final.pt')
    for path in files.values():
        write(output / path)
    write(output / 'pipeline_result.json', json.dumps(files))


class GeometryJobTests(unittest.TestCase):
    def test_automatic_jobs_freeze_lidar_units_and_do_not_require_a_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts(root)
            artifacts(root, 'model_02')
            save_geometry_settings([dict(class_name='Dewa', model_id=mdl, lidar_unit=unit,
                alignment_mode='automatic') for mdl, unit in [('model_01', 'm'), ('model_02', 'mm')]], root)
            jobs, _ = prepare_jobs(dict(action='evaluate', evaluation_method='both', geometry_settings_required=True), root)
            self.assertEqual(len(jobs), 4)
            self.assertEqual([j['geometry_options']['lidar_unit'] for j in jobs], ['m', 'm', 'mm', 'mm'])
            for job in jobs:
                self.assertIsNone(job['scale_factor'])
                command = build_command(job, root)
                self.assertIn('--auto_align', command)
                self.assertNotIn('--scale_factor', command)
                self.assertEqual(command[command.index('--lidar_unit') + 1], job['geometry_options']['lidar_unit'])
            save_geometry_settings([dict(class_name='Dewa', model_id='model_01', lidar_unit='cm', alignment_mode='automatic')], root)
            self.assertEqual(jobs[0]['geometry_options']['lidar_unit'], 'm')
            self.assertEqual(get_project_status(root)['Dewa']['model_01']['geometry_settings']['lidar_unit'], 'cm')

    def test_unit_settings_are_atomic_explicit_and_preserve_other_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = dict(class_name='Dewa', model_id='model_01', lidar_unit='m', alignment_mode='automatic')
            save_geometry_settings([valid], root)
            for unit in ('', None, 'feet', True):
                with self.subTest(unit=unit), self.assertRaises(ValueError):
                    save_geometry_settings([{**valid, 'lidar_unit': 'mm'},
                        {**valid, 'model_id': 'model_02', 'lidar_unit': unit}], root)
                self.assertEqual(read_geometry_settings(root)['Dewa']['model_01']['lidar_unit'], 'm')
            save_geometry_settings([{**valid, 'model_id': 'model_02', 'lidar_unit': 'cm'}], root)
            save_geometry_settings([dict(class_name='Dewa', model_id='model_01', clear=True)], root)
            self.assertNotIn('model_01', read_geometry_settings(root)['Dewa'])
            self.assertEqual(read_geometry_settings(root)['Dewa']['model_02']['lidar_unit'], 'cm')

    def test_new_dashboard_never_falls_back_to_old_manual_scale_without_lidar_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts(root)
            save_geometry_scales([dict(class_name='Dewa', model_id='model_01', scale_factor=1)], root)
            jobs, skipped = prepare_jobs(dict(action='evaluate', class_name='Dewa', model_id='model_01', geometry_settings_required=True), root)
            self.assertFalse(jobs)
            self.assertIn('satuan', skipped[0])
            with self.assertRaises(ValueError):
                prepare_jobs(dict(action='evaluate', geometry_options=dict(lidar_unit='m', alignment_mode='automatic'), apply_icp=False), root)

    def test_bulk_geometry_uses_each_model_scale_and_freezes_values_in_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_workspace(root)
            artifacts(root)
            artifacts(root, 'model_02')
            entries = [dict(class_name='Dewa', model_id='model_01', scale_factor=0.02),
                       dict(class_name='Dewa', model_id='model_02', scale_factor=0.07)]
            save_geometry_scales(entries, root)
            jobs, skipped = prepare_jobs(dict(action='evaluate', evaluation_method='both', apply_icp=False), root)
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len(skipped), 32)
            self.assertEqual({job['evaluation_method'] for job in jobs}, {'3DGS', 'SuGaR'})
            self.assertEqual([job['scale_factor'] for job in jobs], [0.02, 0.02, 0.07, 0.07])
            for job in jobs:
                command = build_command(job, root)
                self.assertTrue(command[2].endswith('05_evaluate_comparison.py'))
                self.assertEqual(command[command.index('--scale_factor') + 1], str(job['scale_factor']))
                self.assertIn('--no_icp', command)
                self.assertNotIn('batch_run_all.py', ' '.join(command))
            save_geometry_scales([dict(class_name='Dewa', model_id='model_01', scale_factor=0.99)], root)
            self.assertEqual(jobs[0]['scale_factor'], 0.02)
            self.assertEqual(get_project_status(root)['Dewa']['model_01']['geometry_scale_factor'], 0.99)

    def test_geometry_reports_missing_inputs_instead_of_assuming_unit_scale(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts(root)
            jobs, skipped = prepare_jobs(dict(action='evaluate', class_name='Dewa', model_id='model_01', evaluation_method='both'), root)
            self.assertEqual(jobs, [])
            self.assertEqual(len(skipped), 2)
            self.assertTrue(all('kalibrasi' in reason for reason in skipped))
            with self.assertRaises(ValueError):
                prepare_jobs(dict(action='evaluate', evaluation_method='both', scale_factor=1), root)
            for scale in (0, -1, True, 'nan', 'inf', None):
                with self.subTest(scale=scale), self.assertRaises(ValueError):
                    prepare_jobs(dict(action='evaluate', class_name='Dewa', model_id='model_01', scale_factor=scale), root)

    def test_calibration_updates_are_validated_before_any_existing_value_is_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_geometry_scales([dict(class_name='Dewa', model_id='model_01', scale_factor=0.02)], root)
            for bad in ([dict(class_name='Dewa', model_id='model_02', scale_factor='nan')],
                        [dict(class_name='../Dewa', model_id='model_01', scale_factor=1)],
                        [dict(class_name='Dewa', model_id='model_02', scale_factor=True)]):
                with self.assertRaises(ValueError):
                    save_geometry_scales([dict(class_name='Dewa', model_id='model_01', scale_factor=0.5), *bad], root)
                self.assertEqual(read_geometry_scales(root)['Dewa']['model_01'], 0.02)

    def test_bulk_visual_only_enqueues_existing_method_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_workspace(root)
            artifacts(root)
            jobs, skipped = prepare_jobs(dict(action='evaluate_visual', evaluation_method='both'), root)
            self.assertEqual(len(jobs), 2)
            self.assertEqual(len(skipped), 34)
            for job in jobs:
                command = build_command(job, root)
                self.assertTrue(command[2].endswith('06_evaluate_rendering.py'))
                self.assertNotIn('03_train_3dgs.py', ' '.join(command))
                self.assertNotIn('04_train_sugar.py', ' '.join(command))

    def test_geometry_completion_requires_current_source_ground_truth_and_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts(root)
            source = root / 'outputs/3dgs/Dewa/model_01/point_cloud/iteration_30000/point_cloud.ply'
            gt = root / 'datasets/Dewa/model_01/ground_truth/lidar.ply'
            report = dict(model_signature=file_signature(source), gt_signature=file_signature(gt),
                          scale_factor_to_meters=0.02, chamfer_distance_cm=1.2)
            report.update({f'{name}_at_{threshold}cm_pct': 90 for threshold in (2, 5)
                           for name in ('precision', 'recall', 'f1')})
            write(root / 'outputs/3dgs/Dewa/model_01/geometry/metrics.json', json.dumps(report))
            self.assertTrue(geometry_result_current(root, 'Dewa', 'model_01', '3DGS', 0.02))
            self.assertFalse(geometry_result_current(root, 'Dewa', 'model_01', '3DGS', 0.03))
            write(gt, 'different ground truth contents')
            self.assertFalse(geometry_result_current(root, 'Dewa', 'model_01', '3DGS', 0.02))

    def test_http_bulk_geometry_saves_scales_then_enqueues_serial_worker_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initialize_workspace(root)
            artifacts(root)
            queue = JobQueue(root, lambda job: self.fail('The test queue must remain paused'), start_worker=False)
            queue.set_paused(True)
            server = HTTPServer(('127.0.0.1', 0), DashboardHandler)
            server.project_root, server.job_queue = root, queue
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = 'http://127.0.0.1:' + str(server.server_port)
            def request(path, payload):
                req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    return json.load(response)
            try:
                request('/api/geometry-scales', {'scales': [dict(class_name='Dewa', model_id='model_01', scale_factor=0.02)]})
                self.assertEqual(queue.snapshot()['jobs'], [])
                first = request('/api/run', dict(action='evaluate', evaluation_method='both'))
                self.assertEqual(len(first['added']), 2)
                self.assertEqual(len(first['skipped']), 34)
                self.assertFalse(queue.snapshot()['is_running'])
                second = request('/api/run', dict(action='evaluate', evaluation_method='both'))
                self.assertEqual(len(second['duplicates']), 2)
                self.assertEqual(len(queue.snapshot()['jobs']), 2)
                queue.cancel_pending()
                result = request('/api/geometry-settings', {'settings': [dict(class_name='Dewa',
                    model_id='model_01', lidar_unit='cm', alignment_mode='automatic')]})
                self.assertEqual(result['settings']['Dewa']['model_01']['lidar_unit'], 'cm')
                automatic = request('/api/run', dict(action='evaluate', evaluation_method='both',
                                                   geometry_settings_required=True))
                self.assertEqual(len(automatic['added']), 2)
                for job in queue.snapshot()['jobs'][-2:]:
                    self.assertEqual(job['geometry_options']['alignment_mode'], 'automatic')
                    self.assertIn('--auto_align', build_command(job, root))
                self.assertFalse(queue.snapshot()['is_running'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
                queue.close()


if __name__ == '__main__':
    unittest.main()
