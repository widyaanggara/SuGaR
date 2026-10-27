"""Resolve mesh/checkpoint paths without relying on a working directory."""
from pathlib import Path


def refined_assets(refined_model_path, scene_path, coarse_mesh_path=None):
    checkpoint = Path(refined_model_path).resolve()
    folder_name = checkpoint.parent.name
    mesh_name = folder_name.split("_normalconsistency")[0].replace("sugarfine", "sugarmesh") + ".ply"
    candidates = [Path(coarse_mesh_path).resolve()] if coarse_mesh_path else [
        checkpoint.parent.parent.parent / "mesh" / mesh_name,
        Path("output/coarse_mesh") / Path(scene_path).name / mesh_name,
    ]
    for mesh in candidates:
        if mesh.is_file() and mesh.stat().st_size > 0:
            return folder_name, mesh.resolve()
    raise FileNotFoundError("Coarse mesh tidak ditemukan: " + ", ".join(map(str, candidates)))
