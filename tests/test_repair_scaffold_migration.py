import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools import repair_scaffold_migration as m


class ScaffoldRepairTest(unittest.TestCase):
    def test_verified_receipts_and_preserved_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / 'old', root / 'new'
            repo = root / 'repo'
            repo.mkdir()
            identity = {'scene': {'scene': str(root / 'scene')}, 'repo': str(repo), 'code': {}, 'adapter': {}, 'iterations': 30000}
            run = Path('ablations/scaffold_anchor/scaffold/Punakawan/model_02/runs') / m.fingerprint(identity)
            anchor, initial = run / 'point_cloud/iteration_30000/point_cloud.ply', run / 'initial_points.ply'
            model = Path('ablations/scaffold_anchor/3dgs/Punakawan/model_02')
            for base in (old, new):
                for rel in (anchor, initial):
                    p = base / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b'original')
                    os.utime(p, ns=(1700000000000000000, 1700000000000000000 if base == old else 1700000001000000000))
            train = {'identity': identity, 'inputs': {}, 'output': str(root / 'outputs' / anchor), 'signature': m.signature(old / anchor)}
            conversion = {'identity': identity, 'inputs': {'anchor_ply': train['signature']}, 'output': str(root / 'outputs' / initial), 'signature': m.signature(old / initial)}
            settings = {'initialization': {'path': str(root / 'outputs' / initial), 'signature': conversion['signature']}, 'colmap_signature': {}, 'white_background': False}
            receipts = {run / 'stages/training.json': train, run / 'stages/conversion.json': conversion, model / 'training_settings.json': settings}
            for base in (old, new):
                for rel, value in receipts.items():
                    path = base / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
            with patch.object(m, 'scene_identity', return_value=identity['scene']), patch.object(m, 'code_identity', return_value={}), patch.object(m, 'colmap_signature', return_value={}), patch.object(m, 'read_training_config', return_value={'eval': True}):
                m.repair(root, old, new)
                self.assertEqual(json.loads((new / model / 'training_settings.json').read_text()), settings)
                (new / initial).write_bytes(b'modified')
                with self.assertRaises(ValueError):
                    m.repair(root, old, new, True)
                (new / initial).write_bytes(b'original')
                m.repair(root, old, new, True)
            for rel, value in receipts.items():
                self.assertEqual(json.loads((old / rel).read_text()), value)
                self.assertEqual(json.loads(next(root.glob('backup-scaffold-migration-*')).joinpath(rel).read_text()), value)
            repaired = json.loads((new / model / 'training_settings.json').read_text())
            self.assertEqual(repaired['initialization'], {'path': str((new / initial).resolve()), 'signature': m.signature(new / initial)})
            repaired['initialization'] = settings['initialization']
            self.assertEqual(repaired, settings)
            for rel in (anchor, initial):
                self.assertEqual((new / rel).read_bytes(), (old / rel).read_bytes())


if __name__ == '__main__':
    unittest.main()
