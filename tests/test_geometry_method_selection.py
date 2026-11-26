"""Selection validation and expansion without GPU or filesystem fixtures."""
import unittest
import json
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from web_dashboard.server import prepare_jobs, build_command


class GeometryMethodSelectionTests(unittest.TestCase):
    def jobs(self, methods, experiment='baseline', group=False):
        with patch('web_dashboard.server.scene_pairs', return_value=[('Dewa', 'model_01')]), \
                patch('web_dashboard.server.read_geometry_scales', return_value={}), \
                patch('web_dashboard.server.read_geometry_settings', return_value={}), \
                patch('web_dashboard.server.nonempty_file', return_value=True), \
                patch('web_dashboard.server.geometry_source_ready', return_value=True):
            return prepare_jobs(dict(action='evaluate', class_name='Dewa', model_id='model_01',
                evaluation_methods=methods, experiment=experiment, group_geometry=group,
                geometry_options={'lidar_unit': 'm', 'alignment_mode': 'automatic'}))

    def test_all_methods_create_individual_jobs(self):
        methods = ['3DGS', 'SuGaR', '2DGS', 'RaDe-GS']
        jobs, skipped = self.jobs(methods)
        self.assertEqual([job['evaluation_method'] for job in jobs], methods)
        self.assertEqual(skipped, [])

    def test_subset_deduplicated(self):
        jobs, _ = self.jobs(['2DGS', 'RaDe-GS', '2DGS'])
        self.assertEqual([job['evaluation_method'] for job in jobs], ['2DGS', 'RaDe-GS'])

    def test_grouped_geometry_creates_one_job_and_frozen_runner_payload(self):
        jobs, _ = self.jobs(['RaDe-GS', '3DGS', 'SuGaR', '2DGS'], 'scaffold_anchor', group=True)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['evaluation_methods'], ['3DGS', 'SuGaR', '2DGS', 'RaDe-GS'])
        command = build_command(jobs[0])
        self.assertTrue(command[2].endswith('evaluate_geometry_model.py'))
        payload = json.loads(command[-1])
        self.assertEqual(payload['experiment'], 'scaffold_anchor')
        self.assertEqual(payload['geometry_options']['lidar_unit'], 'm')

    def test_runner_continues_after_method_failure_and_reports_each_result(self):
        from pipeline_tools.evaluate_geometry_model import run
        jobs, _ = self.jobs(['3DGS', '2DGS'], group=True)
        output = io.StringIO()
        with patch('pipeline_tools.evaluate_geometry_model.subprocess.run',
                   side_effect=[SimpleNamespace(returncode=1), SimpleNamespace(returncode=0)]) as process, \
                redirect_stdout(output):
            self.assertEqual(run(jobs[0]), 1)
        self.assertEqual(process.call_count, 2)
        for call, method in zip(process.call_args_list, ['3DGS', '2DGS']):
            command = call.args[0]
            self.assertEqual(command[command.index('--method') + 1], method)
        self.assertIn('3DGS: GAGAL', output.getvalue())
        self.assertIn('2DGS: BERHASIL', output.getvalue())
        self.assertIn('1 berhasil; 1 gagal', output.getvalue())

    def test_runner_success_when_all_methods_succeed(self):
        from pipeline_tools.evaluate_geometry_model import run
        jobs, _ = self.jobs(['3DGS', '2DGS'], group=True)
        with patch('pipeline_tools.evaluate_geometry_model.subprocess.run',
                   return_value=SimpleNamespace(returncode=0)), redirect_stdout(io.StringIO()):
            self.assertEqual(run(jobs[0]), 0)

    def test_scaffold_geometry_routes_all_methods_without_training_setup(self):
        methods = ['3DGS', 'SuGaR', '2DGS', 'RaDe-GS']
        jobs, skipped = self.jobs(methods, 'scaffold_anchor')
        self.assertEqual(len(jobs), 4)
        self.assertEqual(skipped, [])
        for job in jobs:
            with patch.object(Path, 'read_text', return_value='{"surface_mesh":"mesh/statue.ply"}'):
                command = build_command(job)
            self.assertEqual(command[command.index('--experiment') + 1], 'scaffold_anchor')
            self.assertIn('outputs/ablations/scaffold_anchor/', command[command.index('-p') + 1].replace('\\', '/'))
            self.assertIn('datasets/Dewa/model_01/ground_truth/lidar.ply', command[command.index('-g') + 1].replace('\\', '/'))

    def test_empty_and_invalid_selections_rejected(self):
        for methods in ([], 'all', ['both'], [None], [['3DGS']]):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                self.jobs(methods)


if __name__ == '__main__':
    unittest.main()
