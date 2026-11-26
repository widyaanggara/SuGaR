"""Selection validation and expansion without GPU or filesystem fixtures."""
import unittest
from unittest.mock import patch

from web_dashboard.server import prepare_jobs


class GeometryMethodSelectionTests(unittest.TestCase):
    def jobs(self, methods):
        with patch('web_dashboard.server.scene_pairs', return_value=[('Dewa', 'model_01')]), \
                patch('web_dashboard.server.read_geometry_scales', return_value={}), \
                patch('web_dashboard.server.read_geometry_settings', return_value={}), \
                patch('web_dashboard.server.nonempty_file', return_value=True), \
                patch('web_dashboard.server.geometry_source_ready', return_value=True):
            return prepare_jobs(dict(action='evaluate', class_name='Dewa', model_id='model_01',
                evaluation_methods=methods,
                geometry_options={'lidar_unit': 'm', 'alignment_mode': 'automatic'}))

    def test_all_methods_create_individual_jobs(self):
        methods = ['3DGS', 'SuGaR', '2DGS', 'RaDe-GS']
        jobs, skipped = self.jobs(methods)
        self.assertEqual([job['evaluation_method'] for job in jobs], methods)
        self.assertEqual(skipped, [])

    def test_subset_deduplicated(self):
        jobs, _ = self.jobs(['2DGS', 'RaDe-GS', '2DGS'])
        self.assertEqual([job['evaluation_method'] for job in jobs], ['2DGS', 'RaDe-GS'])

    def test_empty_and_invalid_selections_rejected(self):
        for methods in ([], 'all', ['both'], [None], [['3DGS']]):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                self.jobs(methods)


if __name__ == '__main__':
    unittest.main()
