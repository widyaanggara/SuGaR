"""Recovery safety checks with tiny CPU fixtures; no CUDA/training dependencies."""
import json
import pickle
from pathlib import Path
import tempfile
import unittest

from pipeline_tools.recovery_state import CheckpointStore, StageJournal, configure_recovery
from web_dashboard.job_queue import JobQueue


class RecoveryStoreTests(unittest.TestCase):
    def test_interrupted_write_preserves_last_checkpoint_and_corrupt_latest_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder, {'scene': 'a', 'resolution': 1600}, log=lambda _: None)
            def validate(state, iteration):
                if not state.get('optimizer'):
                    raise ValueError('Missing optimizer')
            payload = {'weights': [1, 2], 'optimizer': {'momentum': [3, 4]}}
            store.save(payload, 1000, pickle.dump)
            def broken(value, stream):
                stream.write(b'partial')
                raise OSError('disk full')
            with self.assertRaises(OSError):
                store.save(payload, 2000, broken)
            self.assertEqual(store.load(lambda p: pickle.loads(p.read_bytes()), validate)[1], 1000)
            store.save(payload, 2000, pickle.dump)
            latest = store.candidates()[0][1]
            latest.write_bytes(b'corrupt')
            state, iteration = store.load(lambda p: pickle.loads(p.read_bytes()), validate)
            self.assertEqual(iteration, 1000)
            self.assertEqual(state, payload)
            # Recovered older checkpoint stays protected while the bad slot is replaced.
            store.save(payload, 1500, pickle.dump)
            self.assertEqual([i for i, _ in store.candidates()], [1500, 1000])

    def test_no_safe_snapshot_raises_instead_of_silently_starting_from_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            store = CheckpointStore(folder, {'scene': 'a'}, log=lambda _: None)
            store.save({'weights_only': True}, 1000, pickle.dump)
            def reject(state, iteration):
                raise ValueError('Optimizer missing')
            with self.assertRaisesRegex(ValueError, 'Tidak ada checkpoint valid'):
                store.load(lambda p: pickle.loads(p.read_bytes()), reject)

    def test_different_configuration_and_fresh_generation_never_mix_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'identity.json'
            old = configure_recovery(path, {'resolution': 1600})
            self.assertEqual(configure_recovery(path, {'resolution': 1600}), old)
            first = CheckpointStore(folder, old)
            first.save({'weights': []}, 20000, pickle.dump)
            fresh = configure_recovery(path, {'resolution': 1600}, resume=False)
            self.assertNotEqual(old, fresh)
            self.assertEqual(CheckpointStore(folder, fresh).candidates(), [])
            changed = configure_recovery(path, {'resolution': 800})
            self.assertNotEqual(changed, fresh)
            self.assertTrue(first.candidates())

    def test_complete_sugar_stages_require_input_configuration_and_texture_dependencies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            obj, mtl, texture = root / 'model.obj', root / 'model.mtl', root / 'model.png'
            obj.write_text('mtllib model.mtl\nv 0 0 0\n')
            mtl.write_text('map_Kd model.png\n')
            texture.write_bytes(b'png')
            journal = StageJournal(root / 'stages', {'resolution': 1600}, log=lambda _: None)
            journal.complete('texture', {'mesh_signature': 1}, obj)
            self.assertEqual(journal.get('texture', {'mesh_signature': 1}), str(obj.resolve()))
            self.assertIsNone(journal.get('texture', {'mesh_signature': 2}))
            self.assertIsNone(StageJournal(root / 'stages', {'resolution': 800}).get('texture', {'mesh_signature': 1}))
            texture.unlink()
            self.assertIsNone(journal.get('texture', {'mesh_signature': 1}))


class PriorityRecoveryTests(unittest.TestCase):
    def test_retry_promotes_existing_pending_job_and_preserves_original_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = JobQueue(folder, lambda _: self.fail('Test must not run a training process'), start_worker=False)
            queue.set_paused(True)
            original = dict(action='full_pipeline', class_name='Dewa', model_id='model_01',
                            gpu=0, resume=False, pipeline_start='existing_colmap')
            ids, _ = queue.enqueue([original, {**original, 'model_id': 'model_02'}])
            queue.jobs[0]['status'] = 'INTERRUPTED'
            duplicate, _ = queue.enqueue([original])
            result = queue.retry(ids[0])
            self.assertEqual(result['job_id'], duplicate[0])
            self.assertEqual(len(queue.jobs), 3)
            self.assertTrue(queue.jobs[-1]['resume'])
            self.assertTrue(queue.jobs[-1]['recovery'])
            self.assertEqual(queue.jobs[-1]['pipeline_start'], 'existing_colmap')
            self.assertTrue(queue.paused)
            again = queue.retry(ids[0])
            self.assertEqual(again['job_id'], result['job_id'])
            persisted = json.loads(queue.state_path.read_text())
            self.assertTrue(persisted['jobs'][-1]['recovery'])
            queue.close()

    def test_retry_never_replaces_active_job_or_pending_different_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            queue = JobQueue(folder, lambda _: self.fail('No process should start'), start_worker=False)
            original = dict(action='3dgs', class_name='Dewa', model_id='model_01', gpu=0)
            ids, _ = queue.enqueue([original])
            queue.jobs[0]['status'] = 'FAILED'
            queued, _ = queue.enqueue([{**original, 'gpu': 1}])
            with self.assertRaisesRegex(ValueError, 'pengaturan berbeda'):
                queue.retry(ids[0])
            queue.jobs[-1]['status'] = 'RUNNING'
            result = queue.retry(ids[0])
            self.assertTrue(result['already_running'])
            self.assertFalse(queue.jobs[-1].get('recovery', False))
            self.assertEqual(queue.jobs[-1]['id'], queued[0])
            queue.close()


if __name__ == '__main__':
    unittest.main()
