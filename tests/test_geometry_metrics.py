"""Geometry score definitions and report export, without CUDA/Open3D dependencies."""
import argparse
import ast
import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pipeline_tools.geometry_metrics import metrics_from_distances, GEOMETRY_COLUMNS
from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.dataset_registry import file_signature
from web_dashboard.server import get_metrics_table


class GeometryMetricsTests(unittest.TestCase):
    def test_asymmetric_point_sets_keep_precision_recall_direction_and_unsquared_chamfer(self):
        source = [(0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)]
        target = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        nearest = lambda a, b: [min(math.dist(p, q) for q in b) for p in a]
        scores = metrics_from_distances(nearest(source, target), nearest(target, source))
        self.assertEqual(scores['precision_at_2cm_pct'], 50)
        self.assertAlmostEqual(scores['recall_at_2cm_pct'], 66.6667)
        self.assertAlmostEqual(scores['f1_at_2cm_pct'], 57.1429)
        self.assertAlmostEqual(scores['chamfer_distance_cm'], 54.1667)
        self.assertEqual(scores['source_evaluation_points'], 4)
        self.assertEqual(scores['target_evaluation_points'], 3)

    def test_strict_thresholds_zero_f1_and_perfect_reconstruction(self):
        scores = metrics_from_distances([0, .019, .02, .049, .05], [.01, .02, .03, .1])
        self.assertEqual(scores['precision_at_2cm_pct'], 40)
        self.assertEqual(scores['recall_at_2cm_pct'], 25)
        self.assertEqual(scores['f1_at_2cm_pct'], 30.7692)
        self.assertEqual(scores['precision_at_5cm_pct'], 80)
        self.assertEqual(scores['recall_at_5cm_pct'], 75)
        self.assertEqual(scores['f1_at_5cm_pct'], 77.4194)
        perfect = metrics_from_distances([0, 0], [0])
        self.assertEqual(perfect['chamfer_distance_cm'], 0)
        self.assertEqual(perfect['f1_at_2cm_pct'], 100)
        self.assertEqual(metrics_from_distances([1], [1])['f1_at_2cm_pct'], 0)
        for source, target in (([], [0]), ([0], []), ([math.nan], [0]), ([0], [math.inf]), ([-.1], [0])):
            with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                metrics_from_distances(source, target)

    def test_legacy_csv_upgrade_preserves_history_and_exports_new_geometry_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = ThesisLogger('Dewa', 'model_01', 'SuGaR', base_dir=root)
            try:
                with logger.csv_file.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['Kelas', 'Metode', 'PSNR (dB)', 'Status', 'Catatan'])
                    writer.writeheader()
                    writer.writerow(dict(Kelas='Dewi', Metode='3DGS', Status='COMPLETED', Catatan='hasil lama', **{'PSNR (dB)': 20}))
                scores = metrics_from_distances([0, .01, .07], [0, .03])
                logger.set_metrics(chamfer_distance_cm=scores['chamfer_distance_cm'])
                logger.set_geometry_metrics(scores)
                logger.finalize()
                rows = get_metrics_table(root)
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['PSNR (dB)'], '20')
                self.assertEqual(rows[0]['Catatan'], 'hasil lama')
                self.assertEqual(rows[0]['F1 @2cm (%)'], '-')
                for key, column in GEOMETRY_COLUMNS.items():
                    self.assertEqual(float(rows[1][column]), scores[key])
                report = json.loads(logger.json_file.read_text(encoding='utf-8'))
                self.assertEqual(report['metrics']['f1_at_5cm_pct'], scores['f1_at_5cm_pct'])
                self.assertIn('Precision / Recall / F1', logger.md_file.read_text(encoding='utf-8'))
                with self.assertRaises(ValueError):
                    logger.set_geometry_metrics({'precision_at_2cm_pct': math.nan})
            finally:
                for handler in logger.logger.handlers:
                    handler.close()

    def test_geometry_command_persists_per_model_json_and_summary_metrics(self):
        path = Path(__file__).resolve().parents[1] / 'pipeline_tools/05_evaluate_comparison.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'model.ply', root / 'lidar.ply'
            source.write_text('source fixture')
            target.write_text('target fixture')
            scores = metrics_from_distances([0, .01, .07], [0, .03])
            compute = Mock(return_value=scores)
            loggers = []
            def factory(**kwargs):
                logger = ThesisLogger(**kwargs, base_dir=root)
                loggers.append(logger)
                return logger
            namespace = dict(argparse=argparse, Path=Path, sys=sys, math=math, json=json,
                ThesisLogger=factory, PROJECT_ROOT=root, compute_chamfer_distance=compute, file_signature=file_signature)
            exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), 'exec'), namespace)
            argv = ['05_evaluate_comparison.py', '--statue_class', 'Dewa', '--model_id', 'model_01',
                    '--method', 'SuGaR', '--model_path', str(source), '--gt_lidar', str(target),
                    '--scale_factor', '0.01', '--no_icp']
            try:
                with patch.object(sys, 'argv', argv):
                    namespace['main']()
                report = json.loads((root / 'outputs/sugar/Dewa/model_01/geometry/metrics.json').read_text())
                self.assertEqual(report['f1_at_2cm_pct'], scores['f1_at_2cm_pct'])
                self.assertEqual(report['scale_factor_to_meters'], .01)
                self.assertEqual(report['gt_signature'], file_signature(target))
                self.assertFalse(compute.call_args.kwargs['apply_icp'])
                self.assertEqual(float(get_metrics_table(root)[0]['Precision @5cm (%)']), scores['precision_at_5cm_pct'])
            finally:
                for logger in loggers:
                    for handler in logger.logger.handlers:
                        handler.close()


if __name__ == '__main__':
    unittest.main()
