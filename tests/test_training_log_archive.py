import tempfile
import unittest
from pathlib import Path

from web_dashboard.training_logs import archived_logs


class TrainingLogArchiveTests(unittest.TestCase):
    def test_archive_survives_missing_queue_and_separates_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'outputs/logs/training_logs'
            folder.mkdir(parents=True)
            names = ['Dewa_model_01_3DGS_20261128_100000',
                     'Dewa_model_01_3DGS_20261128_110000_123456',
                     'Dewa_model_01_3DGS_scaffold_anchor_20261128_120000',
                     'Dewa_model_02_3DGS_20261128_130000']
            for name in names:
                (folder / (name + '.log')).write_text(name + '\n' + 'progress\n' * 400, encoding='utf-8')
            result = archived_logs(tmp, 'Dewa', 'model_01', '3DGS')
            self.assertEqual(len(result['runs']), 2)
            self.assertEqual(result['session'], names[1])
            self.assertEqual(len(result['text'].splitlines()), 401)
            self.assertEqual(archived_logs(tmp, 'Dewa', 'model_01', '3DGS', names[0])['session'], names[0])
            self.assertEqual(archived_logs(tmp, 'Dewa', 'model_01', '3DGS_scaffold_anchor')['session'], names[2])
            for session in (names[2], '../secret'):
                with self.assertRaises(ValueError):
                    archived_logs(tmp, 'Dewa', 'model_01', '3DGS', session)

    def test_missing_and_invalid_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(archived_logs(tmp, 'Dewa', 'model_01', 'SuGaR')['runs'], [])
            with self.assertRaises(ValueError):
                archived_logs(tmp, '../Dewa', 'model_01', 'SuGaR')

    def test_recovers_training_output_from_deleted_queue_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'outputs/logs/training_logs'
            folder.mkdir(parents=True)
            session = 'Dewa_model_01_SuGaR_20261128_100000'
            (folder / (session + '.log')).write_text('Summary only', encoding='utf-8')
            queue = Path(tmp) / 'outputs/queue/logs'
            queue.mkdir(parents=True)
            text = 'Training iteration 7000 loss 0.01\n[INFO] SESI SELESAI: ' + session + '\n'
            (queue / 'job.log').write_text(text, encoding='utf-8')
            (queue / 'wrong.log').write_text('SESI SELESAI: ' + session + '_123456', encoding='utf-8')
            result = archived_logs(tmp, 'Dewa', 'model_01', 'SuGaR')
            self.assertEqual(result['source'], 'queue')
            self.assertEqual(result['text'], text)
