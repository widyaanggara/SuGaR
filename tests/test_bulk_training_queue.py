import tempfile
import unittest
from pathlib import Path

from web_dashboard.server import prepare_jobs


def write(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')


class BulkTrainingQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for model in ('model_01', 'model_02'):
            scene = self.root / 'datasets/Dewa' / model
            write(scene / 'images/a.jpg')
            for name in ('cameras', 'images', 'points3D'):
                write(scene / f'sparse/0/{name}.bin')
        output = self.root / 'outputs/3dgs/Dewa/model_01'
        write(output / 'point_cloud/iteration_7000/point_cloud.ply')
        write(output / 'cameras.json')

    def tearDown(self):
        self.temporary.cleanup()

    def queued(self, **data):
        jobs, skipped = prepare_jobs(data, self.root)
        return [f"{job['class_name']}/{job['model_id']}" for job in jobs], skipped

    def test_bulk_3dgs_queues_every_model_with_colmap(self):
        jobs, skipped = self.queued(action='3dgs', class_name='', model_id='')
        self.assertEqual(jobs, ['Dewa/model_01', 'Dewa/model_02'])
        self.assertTrue(all('COLMAP' in reason for reason in skipped))

    def test_bulk_sugar_requires_3dgs_7k_checkpoint(self):
        jobs, skipped = self.queued(action='sugar', class_name='Dewa', model_id='')
        self.assertEqual(jobs, ['Dewa/model_01'])
        self.assertEqual(skipped, ['Dewa/model_02: checkpoint 3DGS 7k diperlukan'])

    def test_single_model_sugar_keeps_direct_job(self):
        jobs, skipped = self.queued(action='sugar', class_name='Dewa', model_id='model_02')
        self.assertEqual(jobs, ['Dewa/model_02'])
        self.assertEqual(skipped, [])


if __name__ == '__main__':
    unittest.main()
