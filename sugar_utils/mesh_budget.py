"""Independent geometry budgets; zero preserves the extracted resolution."""
from copy import deepcopy


def simplify_mesh(mesh, triangle_target):
    if isinstance(triangle_target, bool) or not isinstance(triangle_target, int) or triangle_target < 0:
        raise ValueError('Triangle target must be a non-negative integer.')
    if mesh is None:
        return None
    if triangle_target == 0 or len(mesh.triangles) <= triangle_target:
        # Cleaning the result must not mutate the predecimation source.
        return deepcopy(mesh)
    return mesh.simplify_quadric_decimation(triangle_target)
