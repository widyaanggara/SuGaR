"""Small CPU-only synthetic geometry checks; no GPU or dataset is loaded."""
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from pipeline_tools.geometry_alignment import (automatic_similarity_alignment,
    checked_points, fit_similarity, transform_points, AlignmentRejected)


class SimilarityTests(unittest.TestCase):
    def test_evaluator_converts_lidar_units_before_registration_and_scoring(self):
        # Minimal in-memory point-cloud adapter: exercise the real evaluator's
        # unit conversion and Sim(3) path, without installing Open3D on the laptop.
        class Cloud:
            def __init__(self, points):
                self.points = np.array(points, dtype=float).copy()
            def scale(self, factor, center):
                self.points = (self.points - center) * factor + center
            def transform(self, matrix):
                self.points = transform_points(self.points, matrix)
            def compute_point_cloud_distance(self, other):
                return cKDTree(other.points).query(self.points)[0]
        path = Path(__file__).resolve().parents[1] / 'pipeline_tools/05_evaluate_comparison.py'
        spec = importlib.util.spec_from_file_location('geometry_evaluator_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        points = np.random.default_rng(12).uniform(size=(100, 3)) * [1, .4, .2]
        target_meters = points * .37 + [4, 5, 6]
        for unit, factor in [('m', 1), ('cm', 100), ('mm', 1000)]:
            fake = SimpleNamespace(utility=SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
                io=SimpleNamespace(read_triangle_mesh=lambda _: SimpleNamespace(has_triangles=lambda: False),
                    read_point_cloud=lambda name: Cloud(points if name == 'source' else target_meters * factor)))
            with self.subTest(unit=unit), patch.object(module, 'o3d', fake, create=True), patch.object(module, 'O3D_AVAILABLE', True):
                report = module.compute_chamfer_distance('source', 'lidar', auto_align=True, scale_factor=None, lidar_unit=unit)
                self.assertEqual(report['chamfer_distance_cm'], 0)
                self.assertEqual(report['f1_at_2cm_pct'], 100)
                self.assertAlmostEqual(report['scale_factor_to_meters'], .37, places=7)
                self.assertEqual(report['lidar_unit'], unit)

    def test_umeyama_recovers_known_scale_rotation_translation_without_reflection(self):
        points = np.random.default_rng(12).normal(size=(100, 3))
        matrix = np.eye(4)
        matrix[:3, :3] = 2.75 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        matrix[:3, 3] = [2, -8, .3]
        fitted = fit_similarity(points, transform_points(points, matrix))
        np.testing.assert_allclose(fitted, matrix, atol=1e-12)
        self.assertGreater(np.linalg.det(fitted[:3, :3]), 0)

    def test_automatic_alignment_recovers_unknown_scale_and_large_rotation(self):
        points = np.random.default_rng(8).normal(size=(400, 3)) * [1, .45, .2]
        angle = 1.32
        matrix = np.eye(4)
        matrix[:3, :3] = .037 * np.array([[np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        matrix[:3, 3] = [5, -2, 8]
        target = transform_points(points, matrix)
        fitted, report = automatic_similarity_alignment(points, target, use_features=False)
        np.testing.assert_allclose(fitted, matrix, atol=1e-7)
        self.assertAlmostEqual(report['scale_factor_to_meters'], .037, places=7)
        self.assertGreater(report['registration_target_overlap'], .95)
        self.assertEqual(report['alignment_review'], 'automatic_requires_visual_review')

    def test_alignment_is_repeatable_and_does_not_modify_original_points(self):
        source = np.random.default_rng(3).uniform(size=(150, 3)) * [1, 2, 4]
        original = source.copy()
        target = source * 2.3 + [3, 2, 1]
        a, _ = automatic_similarity_alignment(source, target, use_features=False)
        b, _ = automatic_similarity_alignment(source, target, use_features=False)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(source, original)

    def test_empty_nonfinite_and_degenerate_clouds_are_rejected(self):
        for points in ([], np.zeros((40, 3)), np.full((40, 3), np.nan),
                       np.column_stack((np.arange(40), np.zeros((40, 2))))):
            with self.subTest(), self.assertRaises(ValueError):
                checked_points(points)

    def test_unrelated_geometry_fails_instead_of_producing_a_score(self):
        rng = np.random.default_rng(9)
        source = rng.normal(size=(350, 3))
        source /= np.linalg.norm(source, axis=1)[:, None]
        target = rng.normal(size=(350, 3)) * [1, .1, .1]
        logs = []
        with self.assertRaisesRegex(AlignmentRejected, 'Penyelarasan otomatis belum memadai') as failure:
            automatic_similarity_alignment(source, target, use_features=False, log=logs.append)
        self.assertTrue(any('[ALIGN HASIL]' in line and 'LiDAR -> rekonstruksi' in line for line in logs))
        self.assertTrue(any('bukan Precision/Recall @2cm' in line for line in logs))
        diagnostics = failure.exception.diagnostics
        self.assertLess(min(diagnostics['registration_source_overlap'], diagnostics['registration_target_overlap']), .35)
        self.assertEqual(diagnostics['status'], 'alignment_rejected')
        self.assertTrue(np.isfinite(diagnostics['alignment_transform']).all())
        self.assertNotIn('chamfer_distance_cm', diagnostics)

    def test_weak_alignment_opt_in_returns_flagged_result(self):
        rng = np.random.default_rng(9)
        source = rng.normal(size=(350, 3))
        source /= np.linalg.norm(source, axis=1)[:, None]
        target = rng.normal(size=(350, 3)) * [1, .1, .1]
        logs = []
        matrix, report = automatic_similarity_alignment(source, target, use_features=False,
                                                        log=logs.append, allow_weak=True)
        self.assertTrue(np.isfinite(matrix).all())
        self.assertFalse(report['alignment_passed'])
        self.assertEqual(report['alignment_review'], 'weak_alignment_forced_not_validated')
        self.assertTrue(any('[ALIGN LEMAH]' in line for line in logs))

    def test_rejected_candidate_saved_separately_without_scores(self):
        path = Path(__file__).resolve().parents[1] / 'pipeline_tools/05_evaluate_comparison.py'
        spec = importlib.util.spec_from_file_location('rejected_evaluator_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class Cloud:
            def __init__(self):
                self.points = np.random.default_rng(12).normal(size=(50, 3))
            def scale(self, factor, center):
                self.points *= factor
            def transform(self, matrix):
                self.points = transform_points(self.points, matrix)
        writes = []
        fake = SimpleNamespace(utility=SimpleNamespace(random=SimpleNamespace(seed=lambda _: None)),
            io=SimpleNamespace(read_triangle_mesh=lambda _: SimpleNamespace(has_triangles=lambda: False),
                read_point_cloud=lambda _: Cloud(),
                write_point_cloud=lambda name, cloud: writes.append(name) or True))
        error = AlignmentRejected('overlap rejected', {'alignment_transform': np.eye(4).tolist(),
            'status': 'alignment_rejected'})
        folder = Path('geometry/alignment_failures/attempt_test')
        logs = []
        with patch.object(module, 'o3d', fake, create=True), patch.object(module, 'O3D_AVAILABLE', True), \
                patch('pipeline_tools.geometry_alignment.automatic_similarity_alignment', side_effect=error), \
                patch.object(Path, 'mkdir'), patch('tempfile.mkdtemp', return_value=str(folder)), \
                patch.object(Path, 'write_text') as metadata, \
                patch.object(module, 'metrics_from_distances') as scoring:
            with self.assertRaisesRegex(AlignmentRejected, 'Diagnostik kandidat'):
                module.compute_chamfer_distance('source', 'lidar', auto_align=True,
                    save_aligned_path='geometry/aligned_reconstruction.ply', log=logs.append)
        self.assertEqual(writes, [str(folder / 'rejected_candidate.ply'), str(folder / 'ground_truth_meters.ply')])
        self.assertIn('alignment_rejected', metadata.call_args.args[0])
        scoring.assert_not_called()
        self.assertTrue(any('[UKURAN]' in line for line in logs))
        self.assertTrue(any('[DIAGNOSTIK]' in line and 'CloudCompare' in line for line in logs))
        self.assertFalse(any('[METRIK]' in line for line in logs))


if __name__ == '__main__':
    unittest.main()
