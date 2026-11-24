import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('repair', Path(__file__).resolve().parents[1] / 'pipeline_tools/repair_migrated_sugar.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class MigrationTest(unittest.TestCase):
    def test_verified_repair_and_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / 'old', root / 'new'
            model = Path('sugar/Dewa/model_01')
            baseline = Path('3dgs/Dewa/model_01/point_cloud/iteration_7000/point_cloud.ply')
            for base in (old, new):
                for rel in (baseline, model / 'mesh.ply', model / 'refined.pt', model / 'surface.ply'):
                    path = base / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'original content')
            os.utime(old / baseline, ns=(1700000000000000000, 1700000000123456700))
            os.utime(new / baseline, ns=(1700000000000000000, 1700000000000000000))
            data = {'baseline_signature': repair.signature(old / baseline), 'coarse_mesh': 'mesh.ply',
                    'refined_checkpoint': 'refined.pt', 'surface_mesh': 'surface.ply'}
            for base in (old, new):
                (base / model / 'pipeline_result.json').write_text(json.dumps(data))
            before = (new / model / 'pipeline_result.json').read_bytes()
            repair.repair(old, new, model)
            self.assertEqual((new / model / 'pipeline_result.json').read_bytes(), before)
            (new / model / 'refined.pt').write_bytes(b'X' * len(b'original content'))
            with self.assertRaises(ValueError):
                repair.repair(old, new, model, True)
            self.assertEqual((new / model / 'pipeline_result.json').read_bytes(), before)
            (new / model / 'refined.pt').write_bytes(b'original content')
            repair.repair(old, new, model, True)
            self.assertEqual(json.loads((new / model / 'pipeline_result.json').read_text())['baseline_signature'], repair.signature(new / baseline))
            self.assertEqual((old / model / 'pipeline_result.json').read_bytes(), before)
            self.assertEqual(next(root.glob('backup-repair-sugar-*/pipeline_result.before.json')).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
