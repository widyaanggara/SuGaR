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


def prepare(source, destination, scene, baseline, gaussians=3, poisson_depth=10):
    """Validate reuse before any training or mutation of an existing result."""
    source, destination, scene, baseline = map(lambda p: Path(p).resolve(),
                                               (source, destination, scene, baseline))
    for protected in (source, scene, baseline):
        if destination == protected or destination.is_relative_to(protected) or protected.is_relative_to(destination):
            raise ValueError('Output optimasi harus terpisah dari hasil sumber dan dataset.')
    if gaussians not in (1, 3, 4, 6) or not 6 <= poisson_depth <= 12:
        raise ValueError('Gaussian/segitiga atau depth Poisson tidak valid.')
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
    region = camera_focus_region(json.loads((baseline / 'cameras.json').read_text(encoding='utf-8')))
    plan = dict(source=str(source), coarse_checkpoint=str(coarse),
                coarse_signature=file_signature(coarse), baseline_signature=signature,
                scene=scene_identity(scene),
                camera_signature=file_signature(baseline / 'cameras.json'),
                settings={key: manifest[key] for key in (
                    'regularization', 'decimation_target', 'refinement_iterations',
                    'white_background', 'image_downscale', 'image_max_size')},
                gaussians_per_triangle=gaussians, region=region, poisson_depth=poisson_depth)
    if plan['settings']['refinement_iterations'] not in (2000, 7000, 15000):
        raise ValueError('Jumlah iterasi sumber tidak didukung.')
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
            quality_options=dict(region=plan['region'], poisson_depth=plan['poisson_depth']))
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
    parser.add_argument('--poisson_depth', type=int, choices=range(6, 13), default=10)
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
        plan = prepare(source, output, scene, baseline, args.gaussians_per_triangle, args.poisson_depth)
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
