"""Run the native scalable extractor with guarded mesh postprocessing."""
import argparse
import importlib.util
from pathlib import Path
import sys


def post_process_mesh(mesh, cluster_to_keep=8):
    import numpy as np
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError('Ekstraksi menghasilkan mesh kosong; periksa kedalaman dan skala scene.')
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    if counts.size == 0:
        raise RuntimeError('Mesh tidak memiliki komponen segitiga.')
    keep = min(max(int(cluster_to_keep), 1), counts.size)
    threshold = max(int(np.sort(counts)[-keep]), 50)
    if counts.max() < threshold:
        raise RuntimeError('Semua komponen mesh memiliki kurang dari 50 segitiga.')
    print(f'[MESH] Postprocess: {counts.size} komponen; pertahankan hingga {keep}, '
          f'ambang {threshold} segitiga.', flush=True)
    mesh.remove_triangles_by_mask(counts[np.asarray(labels)] < threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError('Mesh kosong setelah post-processing.')
    return mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    options, arguments = parser.parse_known_args()
    repo = options.repo.resolve()
    script = repo / 'mesh_extract_tnt.py'
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('togoglab_native_scalable_mesh', script)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.post_process_mesh = post_process_mesh
    # Use the repository's parser/config merger, including checkpoint cfg_args.
    sys.argv = [str(script), *arguments]
    parser = native.ArgumentParser(description=__doc__)
    model = native.ModelParams(parser, sentinel=True)
    pipeline = native.PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--max_depth', default=20.0, type=float)
    parser.add_argument('--voxel_size', default=0.002, type=float)
    parser.add_argument('--num_cluster', default=1, type=int)
    parser.add_argument('--use_depth_filter', action='store_true')
    args = native.get_combined_args(parser)
    native.safe_state(args.quiet)
    native.render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                       args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter)
    for name in ('recon.ply', 'recon_post.ply'):
        path = Path(args.model_path) / name
        mesh = native.o3d.io.read_triangle_mesh(str(path))
        if not path.is_file() or not mesh.has_vertices() or not mesh.has_triangles():
            raise RuntimeError(f'Mesh hilang, kosong, atau tidak terbaca: {path}')
        print(f'[MESH] Valid: {name}; {len(mesh.vertices)} vertices, '
              f'{len(mesh.triangles)} triangles.', flush=True)
        del mesh


if __name__ == '__main__':
    main()
