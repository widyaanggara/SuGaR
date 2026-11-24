import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools import migrate_coarse_resume as migration


class FakeTorch:
    __version__ = '2.0.1'

    @staticmethod
    def load(path, **kwargs):
        with Path(path).open('rb') as stream:
            return pickle.load(stream)

    @staticmethod
    def save(value, stream):
        pickle.dump(value, stream)


class CoarseMigrationTest(unittest.TestCase):
    def test_checkpoint_preservation_and_guardrails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old, new = root / 'old', root / 'new'
            model = Path('ablations/scaffold_anchor/sugar/Punakawan/model_02')
            baseline = Path('ablations/scaffold_anchor/3dgs/Punakawan/model_02/point_cloud/iteration_7000/point_cloud.ply')
            for base in (old, new):
                path = base / baseline
                path.parent.mkdir(parents=True)
                path.write_bytes(b'baseline')
            os.utime(old / baseline, ns=(1700000000000000000, 1700000000123456700))
            os.utime(new / baseline, ns=(1700000000000000000, 1700000000000000000))
            identity = {'generation': 'unchanged', 'context': {'baseline': migration.signature(old / baseline),
                        'scene': {'scene': str(root / 'dataset')}, 'code': {}, 'cameras': {}, 'reused_coarse': None}}
            slot_identity = {'pipeline': identity, 'stage': 'coarse', 'inputs': {}, 'torch': '2.0.1', 'numpy': '1.26.2'}
            key = migration.fingerprint(slot_identity)
            state = {k: {'value': [1, 2, 3]} for k in ('model', 'optimizer', 'rng', 'training', 'camera_order', 'camera_offset', 'buffers')}
            for base in (old, new):
                recovery = base / model / 'recovery'
                slot = recovery / 'checkpoints/coarse' / key
                slot.mkdir(parents=True)
                migration.write_json(recovery / 'identity.json', identity)
                for name, iteration in (('a', 7000), ('b', 7500)):
                    with (slot / (name + '.pt')).open('wb') as stream:
                        FakeTorch.save({'schema': 1, 'fingerprint': key, 'iteration': iteration, 'state': state}, stream)
                    os.utime(slot / (name + '.pt'), ns=(1700000000000000000, 1700000000000000000))
                    migration.write_json(slot / (name + '.json'), {'identity': slot_identity, 'fingerprint': key,
                        'iteration': iteration, 'signature': migration.signature(slot / (name + '.pt'))})
            old_identity = (old / model / 'recovery/identity.json').read_bytes()
            with patch.object(migration, 'scene_identity', return_value=identity['context']['scene']), patch.object(migration, 'code_identity', return_value={}):
                migration.migrate(old, new, model, root)
                self.assertEqual((new / model / 'recovery/identity.json').read_bytes(), old_identity)
                baseline_bytes = (new / baseline).read_bytes()
                (new / baseline).write_bytes(b'changed!')
                with self.assertRaises(ValueError):
                    migration.migrate(old, new, model, root, True, (FakeTorch, '1.26.2'))
                (new / baseline).write_bytes(baseline_bytes)
                migration.migrate(old, new, model, root, True, (FakeTorch, '1.26.2'))
            result = json.loads((new / model / 'recovery/identity.json').read_text())
            expected = copy.deepcopy(identity)
            expected['context']['baseline'] = migration.signature(new / baseline)
            self.assertEqual(result, expected)
            changed_slot_identity = copy.deepcopy(slot_identity)
            changed_slot_identity['pipeline'] = expected
            result_dir = new / model / 'recovery/checkpoints/coarse' / migration.fingerprint(changed_slot_identity)
            for name, iteration in (('a', 7000), ('b', 7500)):
                envelope = FakeTorch.load(result_dir / (name + '.pt'))
                self.assertEqual(envelope['state'], state)
                self.assertEqual(envelope['iteration'], iteration)
                self.assertEqual(envelope['fingerprint'], result_dir.name)
                rel = model / 'recovery/checkpoints/coarse' / key / (name + '.pt')
                self.assertEqual((new / rel).read_bytes(), (old / rel).read_bytes())
            self.assertEqual((old / model / 'recovery/identity.json').read_bytes(), old_identity)


if __name__ == '__main__':
    unittest.main()
