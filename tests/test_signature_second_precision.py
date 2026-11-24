import json
import os
import tempfile
import unittest
from pathlib import Path

from pipeline_tools.dataset_registry import file_signature
from pipeline_tools.recovery_state import configure_recovery


class SecondPrecisionSignatureTest(unittest.TestCase):
    """Outputs copied to a 9p mount lose sub-second mtimes; old records must still match."""

    def test_old_nanosecond_record_matches_copied_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'point_cloud.ply'
            path.write_bytes(b'abc')
            os.utime(path, ns=(1794855837_000000000, 1794855837_000000000))
            stored = json.loads(json.dumps({'size': 3, 'mtime_ns': 1794855837_084575927}))
            self.assertEqual(stored, file_signature(path))
            self.assertFalse(stored != file_signature(path))
            self.assertEqual({'baseline': stored}, {'baseline': file_signature(path)})
            self.assertNotEqual({'size': 4, 'mtime_ns': 1794855837_084575927}, file_signature(path))
            self.assertNotEqual({'size': 3, 'mtime_ns': 1794855838_084575927}, file_signature(path))

    def test_recovery_identity_survives_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'point_cloud.ply'
            path.write_bytes(b'abc')
            identity_path = Path(folder) / 'identity.json'
            identity_path.write_text(json.dumps({'generation': 'old', 'context': {
                'baseline': {'size': 3, 'mtime_ns': 1794855837_084575927}}}), encoding='utf-8')
            os.utime(path, ns=(1794855837_000000000, 1794855837_000000000))
            identity = configure_recovery(identity_path, {'baseline': file_signature(path)})
            self.assertEqual(identity['generation'], 'old')


if __name__ == '__main__':
    unittest.main()
