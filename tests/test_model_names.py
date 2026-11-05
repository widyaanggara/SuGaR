"""Display name persistence must never rename datasets or change pipeline IDs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline_tools.init_dataset_folders import initialize_workspace
from pipeline_tools.model_names import read_model_names, save_model_name
from web_dashboard.server import build_command, get_metrics_table, get_project_status


class ModelNameTests(unittest.TestCase):
    def test_names_survive_reload_and_initialization_without_moving_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_workspace(root)
            photo = root / 'datasets/Dewa/model_01/input/a.jpg'
            photo.write_bytes(b'original image')
            result = save_model_name('Dewa', 'model_01', '  Ganesha  ', root)
            self.assertEqual(result['display_name'], 'Ganesha')
            save_model_name('Dewi', 'model_01', 'Dewi Saraswati', root)
            initialize_workspace(root)
            status = get_project_status(root)
            self.assertEqual(status['Dewa']['model_01']['display_name'], 'Ganesha')
            self.assertEqual(status['Dewi']['model_01']['display_name'], 'Dewi Saraswati')
            self.assertEqual(status['Dewa']['model_02']['display_name'], 'Model 02')
            self.assertEqual(status['Dewa']['model_01']['images_count'], 1)
            self.assertEqual(status['Dewa']['model_01']['input_path'], 'datasets/Dewa/model_01/input/')
            self.assertEqual(photo.read_bytes(), b'original image')
            self.assertFalse((root / 'datasets/Dewa/Ganesha').exists())
            command = build_command(dict(action='colmap', class_name='Dewa', model_id='model_01'), root)
            self.assertEqual(command[command.index('--model_id') + 1], 'model_01')
            self.assertNotIn('Ganesha', command)
            self.assertEqual(read_model_names(root)['Dewa']['model_01'], 'Ganesha')

    def test_reset_and_benchmark_names(self):
        with tempfile.TemporaryDirectory() as directory:
            save_model_name('Dewa', 'model_01', 'Ganesha', directory)
            result = save_model_name('Dewa', 'model_01', '', directory)
            self.assertEqual(result['display_name'], 'Model 01')
            self.assertEqual(result['custom_name'], '')
            self.assertEqual(read_model_names(directory), {})
            save_model_name('TanksAndTemples', 'tanks_and_temples_1', 'Ignatius', directory)
            self.assertEqual(get_project_status(directory)['TanksAndTemples']['tanks_and_temples_1']['display_name'], 'Ignatius')

    def test_invalid_names_and_targets_never_replace_existing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_model_name('Dewa', 'model_01', 'Ganesha', root)
            path = root / 'datasets/model_names.json'
            original = path.read_bytes()
            for cls, mdl, name in [('', 'model_01', 'Name'), ('../Dewa', 'model_01', 'Name'),
                                   ('Dewa', '', 'Name'), ('Dewa', '../model_01', 'Name'),
                                   ('Dewa', 'model_01', None), ('Dewa', 'model_01', 'a' * 81),
                                   ('Dewa', 'model_01', 'two\nlines')]:
                with self.subTest(name=name, cls=cls, mdl=mdl), self.assertRaises(ValueError):
                    save_model_name(cls, mdl, name, root)
                self.assertEqual(path.read_bytes(), original)

    def test_failed_atomic_replace_retains_previous_name(self):
        with tempfile.TemporaryDirectory() as directory:
            save_model_name('Dewa', 'model_01', 'Ganesha', directory)
            with patch.object(Path, 'replace', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_model_name('Dewa', 'model_01', 'Replacement', directory)
            self.assertEqual(read_model_names(directory)['Dewa']['model_01'], 'Ganesha')
            self.assertFalse((Path(directory) / 'datasets/model_names.tmp').exists())

    def test_invalid_metadata_is_reported_instead_of_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'datasets/model_names.json'
            path.parent.mkdir()
            for text in ('not json', json.dumps([]), json.dumps({'Dewa': []})):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    save_model_name('Dewa', 'model_01', 'Ganesha', directory)
                self.assertEqual(path.read_text(encoding='utf-8'), text)

    def test_metrics_use_current_names_while_retaining_original_ids_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'outputs/logs/evaluation_reports/summary_comparison.csv'
            path.parent.mkdir(parents=True)
            original = 'Kelas,Model ID,Metode,PSNR (dB)\nDewa,model_01,SuGaR,24.5\n'
            path.write_text(original, encoding='utf-8')
            save_model_name('Dewa', 'model_01', 'Ganesha', directory)
            row = get_metrics_table(directory)[0]
            self.assertEqual(row['Nama Model'], 'Ganesha')
            self.assertEqual(row['Model ID'], 'model_01')
            self.assertEqual(row['PSNR (dB)'], '24.5')
            self.assertEqual(path.read_text(encoding='utf-8'), original)


if __name__ == '__main__':
    unittest.main()
