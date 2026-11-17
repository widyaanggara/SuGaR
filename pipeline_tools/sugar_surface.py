"""Export refined surface geometry without the expensive UV texture bake."""
from pathlib import Path


def export_refined_surface(checkpoint, destination):
    import numpy as np
    import torch
    import open3d as o3d
    state = torch.load(str(checkpoint), map_location='cpu')['state_dict']
    vertices = state['_points'].detach().cpu().numpy()
    triangles = state['_surface_mesh_faces'].detach().cpu().numpy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError('Geometri refined SuGaR tidak valid.')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + '.tmp.ply')
    if not o3d.io.write_triangle_mesh(str(temporary), mesh):
        raise RuntimeError('Gagal mengekspor permukaan refined SuGaR.')
    temporary.replace(destination)
    return str(destination)
