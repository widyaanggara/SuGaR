import json
import pickle
import tempfile
import time
import unittest
from pathlib import Path

from pipeline_tools.dataset_registry import file_signature
from pipeline_tools.shrink_sugar_checkpoints import shrink, sugar_outputs, refined_checkpoints

load = lambda path: pickle.loads(Path(path).read_bytes())
save = lambda value, path: Path(path).write_bytes(pickle.dumps(value))


class ShrinkTests(unittest.TestCase):
    def test_drops_optimizer_keeps_model_and_keeps_valid_records_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'outputs/sugar/Dewa/model_01'
            checkpoint = output / 'refined/run/15000.pt'
            checkpoint.parent.mkdir(parents=True)
            model = {'_scales': list(range(100))}
            save({'state_dict': model, 'iteration': 15000, 'optimizer_state_dict': {'m': list(range(1000))}}, checkpoint)
            (output / 'pipeline_result.json').write_text('{}')
            old = file_signature(checkpoint)
            metrics = output / 'evaluation/metrics.json'
            metrics.parent.mkdir()
            metrics.write_text(json.dumps({'checkpoint': 'refined/run/15000.pt', 'checkpoint_signature': old, 'psnr': 23.28}))
            journal = output / 'recovery/stages/abc/refine.json'
            journal.parent.mkdir(parents=True)
            journal.write_text(json.dumps({'output': str(checkpoint), 'signature': old}))
            stale = output / 'recovery/stages/def/refine.json'
            stale.parent.mkdir(parents=True)
            stale.write_text(json.dumps({'output': str(checkpoint), 'signature': {'size': 1, 'mtime_ns': 1}}))
            self.assertEqual(list(sugar_outputs(root)), [output])
            self.assertEqual(refined_checkpoints(output), [checkpoint])
            time.sleep(0.01)
            self.assertTrue(shrink(checkpoint, output, load, save))
            data = load(checkpoint)
            self.assertEqual(data['state_dict'], model)
            self.assertNotIn('optimizer_state_dict', data)
            new = file_signature(checkpoint)
            self.assertLess(new['size'], old['size'])
            report = json.loads(metrics.read_text())
            self.assertEqual(report['checkpoint_signature'], new)
            self.assertEqual(report['psnr'], 23.28)
            self.assertEqual(json.loads(journal.read_text())['signature'], new)
            # A record that was already stale is not made to look valid.
            self.assertEqual(json.loads(stale.read_text())['signature'], {'size': 1, 'mtime_ns': 1})
            self.assertFalse(shrink(checkpoint, output, load, save), 'second run changes nothing')


if __name__ == '__main__':
    unittest.main()
