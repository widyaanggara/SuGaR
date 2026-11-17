import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pipeline_tools.training_vram import write_training_peak
from pipeline_tools.logger_utils import ThesisLogger


class TrainingVramTests(unittest.TestCase):
    def test_child_peak_reaches_csv_even_without_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'training.json'
            torch = MagicMock()
            torch.cuda.is_initialized.return_value = True
            torch.cuda.max_memory_allocated.return_value = 5 * 1024**3
            torch.cuda.max_memory_reserved.return_value = 7 * 1024**3
            with patch.dict(os.environ, {'TOGOGLAB_TRAINING_VRAM_REPORT': str(report)}), \
                 patch.dict(sys.modules, {'torch': torch}):
                write_training_peak()
            logger = ThesisLogger('Dewa', 'model_01', '2DGS', folder)
            with patch.dict(sys.modules, {'torch': None}):
                self.assertTrue(logger.record_child_vram(report, os.getpid()))
                logger.finalize()
            self.assertEqual(logger.data['vram_measurement'], 'torch_peak_allocated')
            self.assertEqual(logger.data['peak_reserved_vram_gb'], 7)
            with logger.csv_file.open(encoding='utf-8', newline='') as stream:
                self.assertEqual(float(next(csv.DictReader(stream))['Peak VRAM (GB)']), 5)
            for handler in logger.logger.handlers:
                handler.close()

    def test_missing_or_wrong_process_report_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            logger = ThesisLogger('Dewa', 'model_01', 'RaDe-GS', folder)
            path = Path(folder) / 'report.json'
            self.assertFalse(logger.record_child_vram(path, 123))
            path.write_text(json.dumps(dict(pid=456, measurement='torch_peak_allocated',
                                           peak_allocated_bytes=1234, peak_reserved_bytes=2345)))
            self.assertFalse(logger.record_child_vram(path, 123))
            self.assertEqual(logger.data['vram_measurement'], 'unavailable')
            for handler in logger.logger.handlers:
                handler.close()

    def test_writer_does_not_initialize_cuda(self):
        torch = MagicMock()
        torch.cuda.is_initialized.return_value = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.json'
            with patch.dict(os.environ, {'TOGOGLAB_TRAINING_VRAM_REPORT': str(path)}), \
                 patch.dict(sys.modules, {'torch': torch}):
                write_training_peak()
            self.assertFalse(path.exists())
            torch.cuda.max_memory_allocated.assert_not_called()


if __name__ == '__main__':
    unittest.main()
