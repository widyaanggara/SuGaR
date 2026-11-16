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
        for changes in ({'class_name': ''}, {'model_id': '../model_01'}, {'confirmation': ''}):
            with self.assertRaises(ValueError):
                delete_model_data(self.queue, {**self.data, **changes})
            self.assertTrue(photo.exists())
        self.queue.active_id = 'running-other-model'
        with self.assertRaises(ValueError):
            delete_model_data(self.queue, self.data)
        self.assertTrue(photo.exists())

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
