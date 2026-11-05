"""Small CPU-only synthetic geometry checks; no GPU or dataset is loaded."""
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from pipeline_tools.geometry_alignment import (automatic_similarity_alignment,
    checked_points, fit_similarity, transform_points)


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
        with self.assertRaisesRegex(ValueError, 'Penyelarasan otomatis belum memadai'):
            automatic_similarity_alignment(source, target, use_features=False)


if __name__ == '__main__':
    unittest.main()
