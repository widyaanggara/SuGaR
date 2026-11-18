"""Rebuild focused SuGaR geometry and refine appearance from an existing coarse model.

Keeps the background for full-image evaluation. Writes to a separate directory;
never replaces the baseline or its research summary. Quality gains require GPU validation.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tools.dataset_registry import file_signature, scene_pairs
from pipeline_tools.gpu_lock import gpu_lock
from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.recovery_state import atomic_json, scene_identity
from pipeline_tools.run_settings import training_matches


def module(name):
    spec = importlib.util.spec_from_file_location('quality_' + name, ROOT / 'pipeline_tools' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


# Pure SuGaR: every Gaussian stays bound to the mesh. Background blur is reduced only through
# SuGaR's own capacity: a finer background Poisson mesh, more triangles and more Gaussians per triangle.
QUALITY_DEFAULTS = dict(gaussians_per_triangle=3, refinement_iterations=15000,
                        foreground_triangles=0, background_triangles=2000000, poisson_depth=11,
                        statue_focus=True, statue_poisson_depth=11)


def job_settings(raw=None):
    """Validate dashboard/CLI values with one set of limits."""
    if raw is not None and not isinstance(raw, dict):
        raise ValueError('Pengaturan optimasi SuGaR harus berupa objek.')
    unknown = set(raw or {}) - set(QUALITY_DEFAULTS)
    if unknown:
        raise ValueError('Pengaturan optimasi SuGaR tidak dikenal: ' + ', '.join(sorted(unknown)))
    result = {**QUALITY_DEFAULTS, **(raw or {})}
    integers = ('gaussians_per_triangle', 'refinement_iterations', 'foreground_triangles',
                'background_triangles', 'poisson_depth', 'statue_poisson_depth')
    if any(isinstance(result[k], bool) or not isinstance(result[k], int) for k in integers):
        raise ValueError('Pengaturan optimasi SuGaR harus bilangan bulat.')
    if result['gaussians_per_triangle'] not in (1, 3, 4, 6):
        raise ValueError('Gaussian per segitiga harus 1, 3, 4, atau 6.')
    if result['refinement_iterations'] not in (2000, 7000, 15000):
        raise ValueError('Iterasi refinement harus 2000, 7000, atau 15000.')
    if not 6 <= result['poisson_depth'] <= 12 or not 6 <= result['statue_poisson_depth'] <= 12:
        raise ValueError('Poisson depth harus 6 sampai 12.')
    for key in ('foreground_triangles', 'background_triangles'):
        if not 0 <= result[key] <= 20000000:
            raise ValueError('Target segitiga harus 0 (tanpa decimation) sampai 20.000.000.')
    if not isinstance(result['statue_focus'], bool):
        raise ValueError('Fokus patung harus boolean.')
    return result


def output_name(settings):
    """Each configuration gets its own folder, so changing values never collides with an old plan."""
    statue = 's%d' % settings['statue_poisson_depth'] if settings['statue_focus'] else 'snone'
    return ('g{gaussians_per_triangle}_it{refinement_iterations}_fg{foreground_triangles}'
            '_bg{background_triangles}_d{poisson_depth}_{statue}').format(statue=statue, **settings)


def prepare(source, destination, scene, baseline, gaussians=3, poisson_depth=11,
            foreground_triangles=0, background_triangles=2000000, refinement_iterations=15000,
            statue_focus=True, statue_poisson_depth=11):
    """Validate reuse before any training or mutation of an existing result."""
    source, destination, scene, baseline = map(lambda p: Path(p).resolve(),
                                               (source, destination, scene, baseline))
    for protected in (source, scene, baseline):
        if destination == protected or destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('Output optimasi harus terpisah dari hasil sumber dan dataset.')
    if gaussians not in (1, 3, 4, 6) or not 6 <= poisson_depth <= 12:
        raise ValueError('Gaussian/segitiga atau depth Poisson tidak valid.')
    if refinement_iterations not in (2000, 7000, 15000):
        raise ValueError('Iterasi refinement harus 2000, 7000, atau 15000.')
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (foreground_triangles, background_triangles)):
        raise ValueError('Target segitiga harus integer >= 0; 0 mempertahankan resolusi asli.')
    options = job_settings(dict(
        gaussians_per_triangle=gaussians, refinement_iterations=refinement_iterations,
        foreground_triangles=foreground_triangles, background_triangles=background_triangles,
        poisson_depth=poisson_depth, statue_focus=statue_focus, statue_poisson_depth=statue_poisson_depth))
    manifest = json.loads((source / 'pipeline_result.json').read_text(encoding='utf-8'))
    if Path(manifest['scene_path']).resolve() != scene:
        raise ValueError('Dataset sumber berbeda dari target.')
    coarse = (source / manifest['coarse_checkpoint']).resolve()
    if not coarse.is_relative_to(source) or not coarse.is_file() or coarse.stat().st_size == 0:
        raise ValueError('Checkpoint coarse hilang, kosong, atau di luar folder sumber.')
    signature = file_signature(baseline / 'point_cloud/iteration_7000/point_cloud.ply')
    if manifest.get('baseline_signature') != signature:
        raise ValueError('Checkpoint baseline berubah; coarse tidak dapat dipakai ulang.')
    if not training_matches(baseline, scene, manifest['image_downscale'], manifest['image_max_size'],
                            manifest['white_background'], use_recorded_initialization=True):
        raise ValueError('Dataset atau protokol training baseline berubah.')
    from pipeline_tools.sugar_focus import camera_focus_region
    from sugar_utils.statue_focus import camera_up
    cameras = json.loads((baseline / 'cameras.json').read_text(encoding='utf-8'))
    region = camera_focus_region(cameras)
    plan = dict(source=str(source), coarse_checkpoint=str(coarse),
                coarse_signature=file_signature(coarse), baseline_signature=signature,
                scene=scene_identity(scene),
                camera_signature=file_signature(baseline / 'cameras.json'),
                settings={key: manifest[key] for key in (
                    'regularization', 'decimation_target', 'refinement_iterations',
                    'white_background', 'image_downscale', 'image_max_size')},
                gaussians_per_triangle=gaussians, region=region, poisson_depth=poisson_depth,
                foreground_triangle_target=foreground_triangles,
                background_triangle_target=background_triangles,
                project_mesh_on_surface_points=False,
                source_refinement_iterations=manifest['refinement_iterations'],
                statue_focus=options['statue_focus'],
                statue_poisson_depth=options['statue_poisson_depth'],
                up=camera_up([c['rotation'] for c in cameras]).tolist())
    # Refinement length is an explicit choice, not inherited from the (often short) baseline run.
    plan['settings']['refinement_iterations'] = refinement_iterations
    receipt = destination / 'optimization_plan.json'
    if receipt.exists():
        if json.loads(receipt.read_text(encoding='utf-8')) != plan:
            raise ValueError('Input/pengaturan berubah; pilih --output_dir baru.')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('Output sudah berisi data tanpa rencana optimasi; pilih folder kosong.')
    return plan


def compare_reports(before, after):
    for key in ('visual_protocol', 'white_background', 'image_downscale', 'image_max_size', 'eval_split_interval'):
        if before[key] != after[key]:
            raise ValueError(f'Protokol evaluasi berbeda: {key}')
    old = {row['image']: row for row in before['per_view']}
    new = {row['image']: row for row in after['per_view']}
    if not old or old.keys() != new.keys() or len(old) != len(before['per_view']) or len(new) != len(after['per_view']):
        raise ValueError('Kamera uji tidak sama atau nama kamera duplikat.')
    return {'before': {k: before[k] for k in ('psnr', 'ssim', 'lpips')},
            'after': {k: after[k] for k in ('psnr', 'ssim', 'lpips')},
            'delta': {k: after[k] - before[k] for k in ('psnr', 'ssim', 'lpips')},
            'per_view': sorted([
                dict(image=name, **{k + '_delta': new[name][k] - old[name][k] for k in ('psnr', 'ssim', 'lpips')})
                for name in old], key=lambda row: row['psnr_delta']),
            'note': 'Full-image held-out evaluation; positive PSNR/SSIM and negative LPIPS deltas indicate improvement.'}


def train_worker(args, plan, scene, baseline, output):
    settings = plan['settings']
    logger = ThesisLogger(args.statue_class, args.model_id, 'SuGaR_quality')
    try:
        module('04_train_sugar').train_sugar_pipeline(
            str(scene), str(baseline), str(output),
            regularization_type=settings['regularization'],
            refinement_time={2000: 'short', 7000: 'medium', 15000: 'long'}[settings['refinement_iterations']],
            n_vertices=settings['decimation_target'], gaussians_per_triangle=plan['gaussians_per_triangle'],
            export_obj=False, gpu=args.gpu, white_background=settings['white_background'], logger=logger,
            image_downscale=settings['image_downscale'], image_max_size=settings['image_max_size'],
            reuse_coarse=plan['coarse_checkpoint'],
            quality_options={key: plan[key] for key in (
                'region', 'poisson_depth', 'foreground_triangle_target',
                'background_triangle_target', 'project_mesh_on_surface_points',
                'statue_focus', 'statue_poisson_depth', 'up')})
        from pipeline_tools.object_mesh import export_object
        export_object(ROOT, args.statue_class, args.model_id, 'sugar', 'baseline',
                      plan['region'], output_override=output)
        logger.finalize(True, export_summary=False)
    except Exception:
        logger.finalize(False, export_summary=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--gaussians_per_triangle', type=int, choices=(1, 3, 4, 6), default=3)
    parser.add_argument('--poisson_depth', type=int, choices=range(6, 13), default=11,
                        help='Depth Poisson background (dan patung bila fokus patung nonaktif).')
    parser.add_argument('--foreground_triangles', type=int, default=0,
                        help='Batas segitiga area patung; default 0: tanpa decimation.')
    parser.add_argument('--background_triangles', type=int, default=2000000,
                        help='Batas segitiga background; 0: tanpa decimation.')
    parser.add_argument('--refinement_iterations', type=int, choices=(2000, 7000, 15000), default=15000,
                        help='Iterasi refinement; default 15000 (default upstream SuGaR).')
    parser.add_argument('--statue_focus', action=argparse.BooleanOptionalAction, default=True,
                        help='Mesh hanya patung: buang bidang tanah dan objek tetangga sebelum Poisson.')
    parser.add_argument('--statue_poisson_depth', type=int, choices=range(6, 13), default=11)
    parser.add_argument('--plan_only', action='store_true')
    parser.add_argument('--train_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU harus >= 0.')
    scene_pairs(args.statue_class, args.model_id)
    suffix = Path(args.statue_class) / args.model_id
    source = ROOT / 'outputs/sugar' / suffix
    baseline = ROOT / 'outputs/3dgs' / suffix
    scene = ROOT / 'datasets' / suffix
    output = (args.output_dir or ROOT / 'outputs/quality/sugar' / suffix).resolve()
    if args.train_worker:
        train_worker(args, json.loads((output / 'optimization_plan.json').read_text(encoding='utf-8')),
                     scene, baseline, output)
        return
    with gpu_lock(args.gpu):
        plan = prepare(source, output, scene, baseline, args.gaussians_per_triangle, args.poisson_depth,
                       args.foreground_triangles, args.background_triangles, args.refinement_iterations,
                       args.statue_focus, args.statue_poisson_depth)
        print(json.dumps(plan, indent=2), flush=True)
        if args.plan_only:
            return
        atomic_json(output / 'optimization_plan.json', plan)
        # Separate processes release all training GPU allocations before rendering.
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--statue_class', args.statue_class, '--model_id', args.model_id,
                        '--gpu', str(args.gpu), '--output_dir', str(output), '--train_worker'], check=True)
        for target in (source, output):
            subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            '--statue_class', args.statue_class, '--model_id', args.model_id,
                            '--gpu', str(args.gpu), '--output_dir', str(target), '--evaluate_worker'], check=True)
        before = json.loads((source / 'evaluation/metrics.json').read_text(encoding='utf-8'))
        after = json.loads((output / 'evaluation/metrics.json').read_text(encoding='utf-8'))
        report = compare_reports(before, after)
        atomic_json(output / 'quality_comparison.json', report)
        print('[QUALITY] Delta:', report['delta'], flush=True)
        print('[QUALITY] Mesh:', output / 'object_mesh/object.ply', flush=True)


def evaluate_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument('--statue_class', required=True)
    parser.add_argument('--model_id', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--evaluate_worker', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.output_dir / 'pipeline_result.json').read_text(encoding='utf-8'))
    logger = ThesisLogger(args.statue_class, args.model_id, 'SuGaR_quality_evaluation')
    try:
        module('06_evaluate_rendering').evaluate_rendering(
            args.statue_class, args.model_id, 'SuGaR', args.gpu, logger,
            manifest['image_downscale'], manifest['image_max_size'], output_override=args.output_dir)
        logger.finalize(True, export_summary=False)
    except Exception:
        logger.finalize(False, export_summary=False)
        raise


if __name__ == '__main__':
    if '--evaluate_worker' in sys.argv:
        evaluate_worker()
    else:
        main()
