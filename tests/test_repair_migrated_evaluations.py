import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools.repair_migrated_evaluations import repair, signature, has_evaluation


class EvaluationRepairTest(unittest.TestCase):
    def test_verified_migration_and_safety(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / 'old', root / 'new'
            model = Path('ablations/scaffold_anchor/sugar/Dewa/model_01')
            for base in (source, target):
                (base / model / 'evaluation').mkdir(parents=True)
                (base / model / 'checkpoint.pt').write_bytes(b'original')
            old, new = source / model, target / model
            os.utime(old / 'checkpoint.pt', ns=(1700000000000000000, 1700000000123456700))
            os.utime(new / 'checkpoint.pt', ns=(1700000000000000000, 1700000000000000000))
            report = {'checkpoint': 'checkpoint.pt', 'checkpoint_signature': signature(old / 'checkpoint.pt'),
                      'per_view': [{'psnr': 25}], 'psnr': 25, 'ssim': .9, 'lpips': .1}
            for base in (old, new):
                (base / 'evaluation/metrics.json').write_text(json.dumps(report))
            before = (new / 'evaluation/metrics.json').read_bytes()
            repair(source, target)
            self.assertEqual((new / 'evaluation/metrics.json').read_bytes(), before)
            (new / 'checkpoint.pt').write_bytes(b'modified')
            with self.assertRaises(ValueError):
                repair(source, target, True)
            self.assertEqual((new / 'evaluation/metrics.json').read_bytes(), before)
            (new / 'checkpoint.pt').write_bytes(b'original')
            repair(source, target, True)
            self.assertTrue(has_evaluation(new))
            repaired = json.loads((new / 'evaluation/metrics.json').read_text())
            repaired['checkpoint_signature'] = report['checkpoint_signature']
            self.assertEqual(repaired, report)
            self.assertEqual((old / 'evaluation/metrics.json').read_bytes(), before)
            self.assertEqual(next(root.glob('backup-repair-evaluation-*/**/metrics.json')).read_bytes(), before)
            repair(source, target, True)  # Reruns preserve already repaired reports.
            report['psnr'] = 99
            (new / 'evaluation/metrics.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                repair(source, target, True)


if __name__ == '__main__':
    unittest.main()
