"""Export an object-only geometry copy without changing the reconstruction."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tools.dataset_registry import file_signature, scene_pairs
from pipeline_tools.external_methods import output_base, mesh_path
from pipeline_tools.object_region import resolve_region, validate_region


def object_status(output, region):
    if (Path(output) / 'object_mesh/error.json').is_file():
        return 'error'
    output = Path(output)
    folder = output / 'object_mesh'
    if region is None:
        return 'unconfigured'
    if (folder / 'error.json').is_file():
        return 'error'
    try:
        saved = json.loads((folder / 'region.json').read_text(encoding='utf-8'))
        source = Path(saved['identity']['source'])
        if not source.resolve().is_relative_to(output.resolve()):
            return 'stale'
        if saved['identity'].get('camera_signature') is not None:
            if saved['identity']['camera_signature'] != file_signature(Path(saved['identity']['cameras'])):
                return 'stale'
        return 'ready' if (saved['identity']['region'] == region
            and saved['identity']['source_signature'] == file_signature(source)
            and saved['output_signature'] == file_signature(folder / 'object.ply')) else 'stale'
    except (OSError, KeyError, ValueError, TypeError):
        return 'missing'


def clean_mesh(mesh, region, o3d):
    import numpy as np
    region = validate_region(region)
    if region is None:
        raise ValueError('Batas patung belum diatur.')
    cropped = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(region['min'], region['max']))
    if not cropped.has_triangles():
        raise ValueError('Batas patung tidak memuat segitiga. Periksa koordinat COLMAP.')
    labels, counts, _ = cropped.cluster_connected_triangles()
    counts = np.asarray(counts)
    cropped.remove_triangles_by_mask(counts[np.asarray(labels)] < region['min_triangles'])
    cropped.remove_duplicated_triangles()
    cropped.remove_degenerate_triangles()
    cropped.remove_unreferenced_vertices()
    if not cropped.has_triangles():
        raise ValueError('Semua komponen terbuang; periksa batas dan ambang fragmen.')
    cropped.compute_vertex_normals()
    return cropped


def export_object(root, cls, mdl, method, experiment, region, output_override=None):
    region = validate_region(region)
    if region is None:
        print(f'[OBJECT] {cls}/{mdl}: batas patung belum diatur; mesh asli dipertahankan.', flush=True)
        return None
    from pipeline_tools.run_settings import colmap_signature
    if ('colmap_signature' in region and region['colmap_signature'] !=
            colmap_signature(Path(root) / 'datasets' / cls / mdl)):
        raise ValueError('COLMAP berubah; simpan ulang batas patung sebelum mengekspor mesh objek.')
    output = Path(output_override) if output_override else output_base(root, experiment) / method / cls / mdl
    automatic = region.get('mode') == 'camera_focus_v1'
    camera_path = output_base(root, experiment) / '3dgs' / cls / mdl / 'cameras.json'
    if automatic and method != 'sugar':
        raise ValueError('Fokus otomatis hanya untuk mesh SuGaR.')
    if method == 'sugar':
        manifest = json.loads((output / 'pipeline_result.json').read_text(encoding='utf-8'))
        source = (output / (manifest.get('surface_mesh') or manifest['textured_mesh'])).resolve()
        if not source.is_relative_to(output.resolve()):
            raise ValueError('Lokasi mesh berada di luar folder hasil.')
    else:
        source = mesh_path(output, method)
    folder = output / 'object_mesh'
    target, receipt = folder / 'object.ply', folder / 'region.json'
    identity = {'version': 1, 'region': region, 'source': str(source),
                'source_signature': file_signature(source)}
    if automatic:
        identity['cameras'] = str(camera_path)
        identity['camera_signature'] = file_signature(camera_path)
    if target.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved.get('identity') == identity and saved.get('output_signature') == file_signature(target):
            print(f'[OBJECT] Mesh patung cocok; dipakai kembali: {target}', flush=True)
            return target
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(source))
    effective_region = region
    if automatic:
        from pipeline_tools.sugar_focus import camera_focus_region
        effective_region = camera_focus_region(json.loads(camera_path.read_text(encoding='utf-8')))
        print(f'[OBJECT] Fokus otomatis SuGaR: {effective_region}', flush=True)
    result = clean_mesh(mesh, effective_region, o3d)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / 'object.tmp.ply'
    if not o3d.io.write_triangle_mesh(str(temporary), result):
        raise RuntimeError('Gagal menyimpan mesh patung.')
    if not o3d.io.read_triangle_mesh(str(temporary)).has_triangles():
        raise RuntimeError('Mesh patung hasil ekspor tidak terbaca.')
    temporary.replace(target)
    payload = {'identity': identity, 'output_signature': file_signature(target),
               'effective_region': effective_region,
               'vertices': len(result.vertices), 'triangles': len(result.triangles),
               'note': 'Geometry-only PLY; original model, textures and evaluation unchanged.'}
    temporary = receipt.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(receipt)
    print(f'[OBJECT] Mesh patung: {target}; {payload["triangles"]} segitiga.', flush=True)
    return target


def export_optional(root, cls, mdl, method, experiment, region, output_override=None):
    """An optional object copy must not abort the remaining training methods."""
    try:
        result = export_object(root, cls, mdl, method, experiment, region, output_override)
    except Exception as exc:
        output = Path(output_override) if output_override else output_base(root, experiment) / method / cls / mdl
        folder = output / 'object_mesh'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'error.json').write_text(json.dumps({'error': str(exc), 'region': region}, indent=2), encoding='utf-8')
        print(f'[WARNING][OBJECT] {cls}/{mdl}/{method}: {exc}. Training/evaluasi scene dilanjutkan.', flush=True)
        return None
    output = Path(output_override) if output_override else output_base(root, experiment) / method / cls / mdl
    (output / 'object_mesh/error.json').unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--method', choices=('sugar', '2dgs', 'radegs'), required=True)
    parser.add_argument('--experiment', choices=('baseline', 'scaffold_anchor'), default='baseline')
    parser.add_argument('--object_region_json')
    parser.add_argument('--optional', action='store_true')
    args = parser.parse_args()
    scene_pairs(args.statue_class, args.model_id)
    region = resolve_region(ROOT / 'datasets' / args.statue_class / args.model_id, args.object_region_json)
    export = export_optional if args.optional else export_object
    export(ROOT, args.statue_class, args.model_id, args.method, args.experiment, region)


if __name__ == '__main__':
    main()
