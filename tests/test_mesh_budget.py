import unittest
from unittest.mock import Mock

from sugar_utils.mesh_budget import simplify_mesh


class MeshBudgetTests(unittest.TestCase):
    def test_full_resolution_is_preserved_and_copy_is_independent(self):
        mesh = Mock(triangles=list(range(20)))
        result = simplify_mesh(mesh, 0)
        mesh.simplify_quadric_decimation.assert_not_called()
        self.assertEqual(len(result.triangles), 20)
        result.triangles.pop()
        self.assertEqual(len(mesh.triangles), 20)

    def test_foreground_and_background_have_independent_budgets(self):
        foreground, background = Mock(triangles=[0]*20), Mock(triangles=[0]*30)
        simplify_mesh(foreground, 0)
        result = simplify_mesh(background, 10)
        foreground.simplify_quadric_decimation.assert_not_called()
        background.simplify_quadric_decimation.assert_called_once_with(10)
        self.assertIs(result, background.simplify_quadric_decimation.return_value)

    def test_no_simplification_below_budget_and_invalid_values_rejected(self):
        mesh = Mock(triangles=[0]*10)
        simplify_mesh(mesh, 20)
        mesh.simplify_quadric_decimation.assert_not_called()
        self.assertIsNone(simplify_mesh(None, 0))
        for target in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                simplify_mesh(mesh, target)
