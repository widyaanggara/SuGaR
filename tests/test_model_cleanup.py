import csv
import json
from pathlib import Path
import tempfile
import unittest

from web_dashboard.job_queue import JobQueue
from web_dashboard.model_cleanup import delete_model_data
from web_dashboard.server import get_project_status


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = JobQueue(self.root, lambda job: [], start_worker=False)
        self.data = dict(class_name='Dewa', model_id='model_01', confirmation='Dewa/model_01')

    def write(self, name, content='data'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def test_complete_cleanup_preserves_other_models_and_survives_restart(self):
        prefixes = ['datasets', 'outputs/3dgs', 'outputs/sugar', 'outputs/2dgs', 'outputs/radegs',
                    'outputs/ablations/scaffold_anchor/scaffold',
                    'outputs/ablations/scaffold_anchor/3dgs',
                    'outputs/ablations/scaffold_anchor/sugar']
        removed = [self.write(p + '/Dewa/model_01/input/a.jpg') for p in prefixes]
        kept = [self.write(p + '/Dewa/model_02/input/a.jpg') for p in prefixes]
        removed.append(self.write('outputs/logs/training_logs/Dewa_model_01_SuGaR_run.json'))
        for name in ('model_names', 'geometry_scales', 'geometry_settings'):
            self.write('datasets/' + name + '.json', json.dumps({'Dewa': {'model_01': 'old', 'model_02': 'keep'}}))
        report = self.write('outputs/logs/evaluation_reports/summary_comparison.csv',
                            'Kelas,Model ID,Metode\nDewa,model_01,3DGS\nDewa,model_01,SuGaR_scaffold_anchor\nDewa,model_02,SuGaR\n')
        ids, _ = self.queue.enqueue([dict(action='full_pipeline', class_name='Dewa', model_id=m)
                                    for m in ('model_01', 'model_02')])
        removed.append(self.write('outputs/queue/logs/' + ids[0] + '.log'))
        delete_model_data(self.queue, self.data)
        self.assertTrue(all(not p.exists() for p in removed))
        self.assertTrue(all(p.read_text() == 'data' for p in kept))
        with report.open() as stream:
            self.assertEqual([r['Model ID'] for r in csv.DictReader(stream)], ['model_02'])
        self.assertNotIn('model_01', report.with_suffix('.md').read_text())
        for name in ('model_names', 'geometry_scales', 'geometry_settings'):
            self.assertEqual(json.loads((self.root / ('datasets/' + name + '.json')).read_text()),
                             {'Dewa': {'model_02': 'keep'}})
        reloaded = JobQueue(self.root, lambda job: [], start_worker=False)
        self.assertEqual([j['model_id'] for j in reloaded.jobs], ['model_02'])
        # Remove artificial metadata for model_02 before validating dashboard state.
        for name in ('geometry_scales', 'geometry_settings'):
            (self.root / ('datasets/' + name + '.json')).write_text('{}')
        self.assertEqual(get_project_status(self.root)['Dewa']['model_01']['overall_status'], 'No Data')
        delete_model_data(self.queue, self.data)  # Empty slots are safe to reset again.

    def test_invalid_target_confirmation_and_active_worker_leave_data_untouched(self):
        photo = self.write('datasets/Dewa/model_01/input/a.jpg')
        for changes in ({'class_name': ''}, {'model_id': '../model_01'}, {'confirmation': ''}, {'scope': 'invalid'}):
            with self.assertRaises(ValueError):
                delete_model_data(self.queue, {**self.data, **changes})
            self.assertTrue(photo.exists())
        self.queue.active_id = 'running-other-model'
        with self.assertRaises(ValueError):
            delete_model_data(self.queue, self.data)
        self.assertTrue(photo.exists())

    def test_method_cleanup_preserves_dataset_other_methods_and_settings(self):
        for scope, label in (('sugar', 'SuGaR'), ('2dgs', '2DGS'), ('radegs', 'RaDe-GS')):
            with self.subTest(scope=scope):
                preserved = [self.write('datasets/Dewa/model_01/input/a.jpg'),
                             self.write('datasets/Dewa/model_01/sparse/0/cameras.bin')]
                for name in ('model_names', 'geometry_scales', 'geometry_settings'):
                    preserved.append(self.write(f'datasets/{name}.json', '{"Dewa":{"model_01":"keep"}}'))
                removed = []
                for method in ('3dgs', 'sugar', '2dgs', 'radegs', 'scaffold'):
                    for base in ('outputs', 'outputs/ablations/scaffold_anchor'):
                        path = self.write(f'{base}/{method}/Dewa/model_01/result.ply')
                        (removed if method == scope else preserved).append(path)
                    log_label = {'sugar': 'SuGaR', '2dgs': '2DGS', 'radegs': 'RaDe-GS'}.get(method, method)
                    path = self.write(f'outputs/logs/training_logs/Dewa_model_01_{log_label}_run.json')
                    (removed if method == scope else preserved).append(path)
                removed.append(self.write(f'outputs/logs/training_logs/Dewa_model_01_{label}_scaffold_anchor_run.json'))
                preserved.append(self.write(f'outputs/{scope}/Dewa/model_02/result.ply'))
                labels = ['3DGS', 'SuGaR', '2DGS', 'RaDe-GS', label + '_scaffold_anchor']
                report = self.write('outputs/logs/evaluation_reports/summary_comparison.csv',
                    'Kelas,Model ID,Metode\n' + ''.join(f'Dewa,model_01,{value}\n' for value in labels))
                self.queue.jobs = []
                ids, _ = self.queue.enqueue([dict(action=action, class_name='Dewa', model_id='model_01')
                                             for action in (scope, 'full_pipeline', '3dgs')])
                delete_model_data(self.queue, {**self.data, 'scope': scope})
                self.assertTrue(all(not path.exists() for path in removed))
                self.assertTrue(all(path.exists() for path in preserved))
                self.assertEqual([job['id'] for job in self.queue.jobs], [ids[2]])
                with report.open() as stream:
                    methods = [row['Metode'] for row in csv.DictReader(stream)]
                self.assertEqual(methods, [value for value in labels if value not in (label, label + '_scaffold_anchor')])

    def test_outside_path_is_rejected_before_deletion(self):
        from unittest.mock import patch
        photo = self.write('datasets/Dewa/model_01/input/a.jpg')
        original = Path.resolve
        target = photo.parents[1]
        def resolve(path, *args, **kwargs):
            return self.root.parent if path == target else original(path, *args, **kwargs)
        with patch.object(Path, 'resolve', resolve):
            with self.assertRaises(ValueError):
                delete_model_data(self.queue, self.data)
        self.assertTrue(photo.exists())


if __name__ == '__main__':
    unittest.main()
