import os
import sys
import json
import gc
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from pipeline_tools.logger_utils import ThesisLogger
from pipeline_tools.gpu_lock import gpu_lock
from pipeline_tools.dataset_registry import has_checkpoint, has_colmap_model, file_signature
from pipeline_tools.run_settings import (add_resolution_arguments, resolution_settings, training_matches,
    sugar_settings, research_presets)
from pipeline_tools.recovery_state import (configure_recovery, scene_identity, code_identity,
                                          StageJournal, fingerprint)


def statue_focus_options(checkpoint_3dgs, n_vertices):
    """Mesh only the statue (ground and neighbours go to the background mesh).

    Same SuGaR extraction otherwise: SuGaR's camera-based foreground/background boxes, upstream
    Poisson depth 10, and the chosen mesh budget for the statue, its surroundings and the background
    each. Vertex projection is off so the statue part keeps its order and object.ply is the statue
    part of the refined mesh itself, not a crop.
    """
    from pipeline_tools.sugar_focus import camera_focus_region
    from sugar_utils.statue_focus import camera_up
    cameras = json.loads((Path(checkpoint_3dgs) / 'cameras.json').read_text(encoding='utf-8'))
    return dict(region=camera_focus_region(cameras), foreground_box='upstream',
                poisson_depth=10, statue_poisson_depth=10,
                foreground_triangle_target=n_vertices, background_triangle_target=n_vertices,
                project_mesh_on_surface_points=False, statue_focus=True,
                up=camera_up([camera['rotation'] for camera in cameras]).tolist())


def previous_coarse(output_dir, checkpoint_3dgs, settings):
    """A finished coarse model from an earlier run of the same scene/settings.

    Coarse optimization does not depend on mesh extraction, so switching the mesh to statue focus
    only reruns mesh extraction and refinement.
    """
    try:
        previous = json.loads((output_dir / 'pipeline_result.json').read_text(encoding='utf-8'))
        candidate = (output_dir / previous['coarse_checkpoint']).resolve()
        baseline = file_signature(Path(checkpoint_3dgs) / 'point_cloud/iteration_7000/point_cloud.ply')
        if (candidate.is_relative_to(output_dir.resolve()) and candidate.is_file() and candidate.stat().st_size
                and previous.get('baseline_signature') == baseline
                and all(previous.get(key) == value for key, value in settings.items())):
            return str(candidate)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def train_sugar_pipeline(scene_path: str,
                         checkpoint_3dgs: str,
                         output_base_dir: str,
                         regularization_type: str = "dn_consistency",
                         refinement_time: str = "short",
                         n_vertices: int = 250000,
                         gaussians_per_triangle: int = 1,
                         export_obj: bool = True,
                         gpu: int = 0,
                         white_background: bool = False,
                         logger: ThesisLogger = None,
                         image_downscale: int = 1,
                         image_max_size: int = 1600,
                         resume: bool = True,
                         quality_options=None,
                         reuse_coarse=None,
                         statue_focus=False):
    """
    Menjalankan pipeline SuGaR lengkap secara teratur dan aman dari masalah memori/path.
    
    Tahapan:
    1. Coarse Surface-Aligned Optimization (dn_consistency)
    2. Poisson Surface Mesh Extraction
    3. Refinement Training (dengan setting short / 2k iterasi agar tidak berhari-hari)
    4. Export Textured Mesh (.obj, .mtl, .png)
    """
    if not has_colmap_model(scene_path) or not has_checkpoint(checkpoint_3dgs, 7000):
        raise FileNotFoundError("Dataset COLMAP dan checkpoint 3DGS iterasi 7000 harus lengkap.")
    sugar_settings(dict(vertices=n_vertices, refinement_time=refinement_time,
        gaussians_per_triangle=gaussians_per_triangle, regularization=regularization_type))
    quality_options = dict(quality_options or {})
    if not training_matches(checkpoint_3dgs, scene_path, image_downscale, image_max_size, white_background,
                            use_recorded_initialization=True):
        raise ValueError('Resolusi, split, background, atau model COLMAP berbeda dari baseline 3DGS. Jalankan Pipeline Lengkap untuk konfigurasi ini.')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['SUGAR_IMAGE_DOWNSCALE'] = str(image_downscale)
    os.environ['SUGAR_IMAGE_MAX_SIZE'] = str(image_max_size)
    os.environ.setdefault('SUGAR_IMAGE_DEVICE', 'cpu')
    gpu = 0  # Physical GPU is now exposed as logical cuda:0.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA tidak tersedia di environment sugar.')
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    from sugar_trainers.coarse_density import coarse_training_with_density_regularization
    from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
    from sugar_trainers.coarse_density_and_dn_consistency import coarse_training_with_density_regularization_and_dn_consistency
    from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
    from sugar_trainers.refine import refined_training
    from sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    scene_path = str(Path(scene_path).resolve())
    checkpoint_3dgs = str(Path(checkpoint_3dgs).resolve())
    if not checkpoint_3dgs.endswith(os.path.sep):
        checkpoint_3dgs += os.path.sep

    output_dir = Path(output_base_dir).resolve()
    manifest_path = output_dir / 'pipeline_result.json'
    if statue_focus and not quality_options:
        try:
            quality_options = statue_focus_options(checkpoint_3dgs, n_vertices)
        except (OSError, ValueError, KeyError) as exc:
            message = f'[WARNING] Fokus patung tidak dapat dipakai ({exc}); mesh seluruh scene seperti sebelumnya.'
            logger.log(message, level='warning') if logger else print(message, flush=True)
        if quality_options and reuse_coarse is None and resume:
            reuse_coarse = previous_coarse(output_dir, checkpoint_3dgs, dict(
                regularization=regularization_type, image_downscale=image_downscale,
                image_max_size=image_max_size, white_background=white_background))
    if manifest_path.exists():
        manifest_path.unlink()
    coarse_dir = output_dir / "coarse"
    mesh_dir = output_dir / "mesh"
    refined_dir = output_dir / "refined"
    textured_dir = output_dir / "textured_mesh"

    for d in [coarse_dir, mesh_dir, refined_dir, textured_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Tentukan jumlah iterasi refinement
    if refinement_time == "short":
        refinement_iterations = 2000
    elif refinement_time == "medium":
        refinement_iterations = 7000
    elif refinement_time == "long":
        refinement_iterations = 15000
    else:
        refinement_iterations = 2000

    identity = configure_recovery(output_dir / 'recovery/identity.json', {
        'scene': scene_identity(scene_path),
        'baseline': file_signature(Path(checkpoint_3dgs) / 'point_cloud/iteration_7000/point_cloud.ply'),
        'cameras': code_identity(Path(checkpoint_3dgs), ['cameras.json']),
        'torch': str(getattr(torch, '__version__', 'unknown')),
        'regularization': regularization_type, 'iterations': refinement_iterations,
        'vertices': n_vertices, 'gaussians_per_triangle': gaussians_per_triangle,
        'image_downscale': image_downscale, 'image_max_size': image_max_size,
        'white_background': white_background,
        'quality_options': quality_options,
        'reused_coarse': file_signature(Path(reuse_coarse)) if reuse_coarse else None,
        'code': code_identity(PROJECT_ROOT, ['sugar_trainers/coarse_density.py',
            'sugar_trainers/coarse_density_and_dn_consistency.py', 'sugar_trainers/coarse_sdf.py',
            'sugar_trainers/refine.py', 'sugar_scene/sugar_model.py', 'sugar_scene/sugar_optimizer.py',
            'sugar_utils/training_recovery.py', 'sugar_utils/mesh_budget.py',
            'sugar_extractors/coarse_mesh.py', 'sugar_extractors/refined_mesh.py'])}, resume)
    journal = StageJournal(output_dir / 'recovery/stages' / fingerprint(identity), identity,
                           enabled=resume, log=logger.log if logger else print)
    def recovery_options(stage, inputs):
        return {'directory': str(output_dir / 'recovery/checkpoints' / stage),
                'identity': {'pipeline': identity, 'stage': stage, 'inputs': inputs}, 'resume': resume}

    if logger:
        logger.log(f"Memulai Pipeline SuGaR untuk Scene: {scene_path}")
        logger.log(f"Regularisasi: {regularization_type}, Refinement: {refinement_iterations}, "
                   f"Parameter decimation: {n_vertices}, Gaussian/segitiga: {gaussians_per_triangle}")
        logger.set_parameters({'effective_sugar_settings': {
            'regularization': regularization_type, 'refinement_iterations': refinement_iterations,
            'decimation_target': n_vertices, 'gaussians_per_triangle': gaussians_per_triangle,
            'image_downscale': image_downscale, 'image_max_size': image_max_size},
            'upstream_reference': research_presets()['source']})

    # ==================== 1. Coarse SuGaR ====================
    if logger:
        logger.start_stage("SuGaR_01_Coarse_Optimization")

    coarse_args = AttrDict({
        'checkpoint_path': checkpoint_3dgs,
        'scene_path': scene_path,
        'iteration_to_load': 7000,
        'output_dir': str(coarse_dir),
        'eval': True,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': gpu,
        'white_background': white_background,
        'recovery': recovery_options('coarse', {}),
    })

    coarse_sugar_path = journal.get('coarse', {})
    if coarse_sugar_path is None and reuse_coarse:
        import shutil
        copied = coarse_dir / 'reused' / Path(reuse_coarse).name
        copied.parent.mkdir(parents=True, exist_ok=True)
        if Path(reuse_coarse).resolve() != copied.resolve():
            shutil.copy2(reuse_coarse, copied)
        coarse_sugar_path = str(copied)
        journal.complete('coarse', {}, coarse_sugar_path)
    coarse_reused = coarse_sugar_path is not None
    if coarse_sugar_path is None:
        if regularization_type == 'sdf':
            coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
        elif regularization_type == 'density':
            coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
        elif regularization_type == 'dn_consistency':
            coarse_sugar_path = coarse_training_with_density_regularization_and_dn_consistency(coarse_args)
        else:
            raise ValueError(f"Unknown regularization type: {regularization_type}")
        journal.complete('coarse', {}, coarse_sugar_path)

    if logger:
        logger.end_stage("SuGaR_01_Coarse_Optimization")
        logger.log(f"Coarse SuGaR tersimpan di: {coarse_sugar_path}")
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 2. Mesh Extraction ====================
    if logger:
        logger.start_stage("SuGaR_02_Mesh_Extraction")

    coarse_mesh_args = AttrDict({
        'scene_path': scene_path,
        'checkpoint_path': checkpoint_3dgs,
        'iteration_to_load': 7000,
        'coarse_model_path': coarse_sugar_path,
        'surface_level': 0.3,
        'decimation_target': n_vertices,
        'project_mesh_on_surface_points': True,
        'mesh_output_dir': str(mesh_dir),
        'bboxmin': None,
        'bboxmax': None,
        'center_bbox': True,
        'gpu': gpu,
        'eval': True,
        'use_centers_to_extract_mesh': False,
        'use_marching_cubes': False,
        'use_vanilla_3dgs': False,
    })
    if quality_options:
        region = quality_options['region']
        if quality_options.get('foreground_box', 'focus') == 'focus':
            coarse_mesh_args.update(bboxmin=str(tuple(region['min'])), bboxmax=str(tuple(region['max'])),
                                    center_bbox=False)
        # Otherwise SuGaR's own camera-based foreground/background boxes are kept.
        coarse_mesh_args.update(poisson_depth=quality_options['poisson_depth'], save_predecimation=True,
                                statue_box=(tuple(region['min']), tuple(region['max'])))
        coarse_mesh_args.update(
            foreground_triangle_target=quality_options.get('foreground_triangle_target', n_vertices),
            background_triangle_target=quality_options.get('background_triangle_target', n_vertices),
            project_mesh_on_surface_points=quality_options.get('project_mesh_on_surface_points', True),
            statue_focus=quality_options.get('statue_focus', False),
            statue_up=quality_options.get('up'),
            statue_poisson_depth=quality_options.get('statue_poisson_depth', quality_options['poisson_depth']))
        if coarse_mesh_args.statue_focus and coarse_mesh_args.statue_up is None:
            raise ValueError('Fokus patung memerlukan arah atas kamera (up).')
    mesh_inputs = {'coarse': file_signature(Path(coarse_sugar_path))}
    coarse_mesh_path = journal.get('mesh', mesh_inputs)
    if coarse_mesh_path is None:
        for stale in ('statue_split.json', 'statue_region.npz'):
            (mesh_dir / stale).unlink(missing_ok=True)  # never pair a new mesh with an old split
        coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]
        journal.complete('mesh', mesh_inputs, coarse_mesh_path)

    if logger:
        logger.end_stage("SuGaR_02_Mesh_Extraction")
        logger.log(f"Poisson surface mesh tersimpan di: {coarse_mesh_path}")
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 3. Refine SuGaR ====================
    if quality_options:
        import open3d as o3d
        mesh_info = o3d.io.read_triangle_mesh(str(coarse_mesh_path))
        counts = dict(vertices=len(mesh_info.vertices), triangles=len(mesh_info.triangles))
        counts['gaussians'] = counts['triangles'] * gaussians_per_triangle
        split_path = mesh_dir / 'statue_split.json'
        if split_path.is_file():
            split = json.loads(split_path.read_text(encoding='utf-8'))
            counts.update(statue_triangles=split['statue_triangles'],
                          background_triangles=counts['triangles'] - split['statue_triangles'])
        print(f'[QUALITY] Mesh used for refinement: {counts}', flush=True)
        from pipeline_tools.recovery_state import atomic_json
        atomic_json(output_dir / 'refinement_geometry.json', counts)
        del mesh_info
    if logger:
        logger.start_stage("SuGaR_03_Refinement")

    refined_args = AttrDict({
        'scene_path': scene_path,
        'checkpoint_path': checkpoint_3dgs,
        'mesh_path': coarse_mesh_path,
        'output_dir': str(refined_dir),
        'iteration_to_load': 7000,
        'normal_consistency_factor': 0.1,
        'gaussians_per_triangle': gaussians_per_triangle,
        'n_vertices_in_fg': n_vertices,
        'refinement_iterations': refinement_iterations,
        'bboxmin': None,
        'bboxmax': None,
        'export_ply': True,
        'eval': True,
        'gpu': gpu,
        'white_background': white_background,
        'recovery': recovery_options('refine', {'mesh': file_signature(Path(coarse_mesh_path))}),
    })
    refine_inputs = {'mesh': file_signature(Path(coarse_mesh_path))}
    refined_sugar_path = journal.get('refine', refine_inputs)
    refine_reused = refined_sugar_path is not None
    if logger:
        logger.set_parameters({'training_reused': coarse_reused and refine_reused})
    if refined_sugar_path is None:
        refined_sugar_path = refined_training(refined_args)
        journal.complete('refine', refine_inputs, refined_sugar_path)

    if logger:
        logger.end_stage("SuGaR_03_Refinement")
        logger.log(f"Refined model tersimpan di: {refined_sugar_path}")
    gc.collect()
    torch.cuda.empty_cache()

    # ==================== 4. Textured Mesh Extraction ====================
    refined_mesh_path = None
    if export_obj:
        if logger:
            logger.start_stage("SuGaR_04_Textured_Mesh_Export")

        refined_mesh_args = AttrDict({
            'scene_path': scene_path,
            'iteration_to_load': 7000,
            'checkpoint_path': checkpoint_3dgs,
            'refined_model_path': refined_sugar_path,
            'coarse_mesh_path': coarse_mesh_path,
            'mesh_output_dir': str(textured_dir),
            'n_gaussians_per_surface_triangle': gaussians_per_triangle,
            'square_size': 8,
            'eval': True,
            'gpu': gpu,
            'postprocess_mesh': False,
            'postprocess_density_threshold': 0.1,
            'postprocess_iterations': 5,
        })
        texture_inputs = {'refined': file_signature(Path(refined_sugar_path)), **refine_inputs}
        refined_mesh_path = journal.get('texture', texture_inputs)
        if refined_mesh_path is None:
            refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)
            journal.complete('texture', texture_inputs, refined_mesh_path)

        if logger:
            logger.end_stage("SuGaR_04_Textured_Mesh_Export")
            logger.log(f"Textured OBJ Mesh tersimpan di: {textured_dir}")

    surface_mesh_path = None
    if not export_obj:
        from pipeline_tools.sugar_surface import export_refined_surface
        if logger:
            logger.start_stage('SuGaR_04_Surface_Export')
        surface_mesh_path = export_refined_surface(refined_sugar_path, output_dir / 'surface/refined_surface.ply')
        if logger:
            logger.end_stage('SuGaR_04_Surface_Export')
            logger.log('Ekspor geometri tanpa baking tekstur; evaluasi tetap merender Gaussian refined.')

    artifacts = {'coarse_checkpoint': coarse_sugar_path, 'coarse_mesh': coarse_mesh_path,
                 'refined_checkpoint': refined_sugar_path, 'textured_mesh': refined_mesh_path,
                 'surface_mesh': surface_mesh_path}
    for key, path in artifacts.items():
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Hasil {key} tidak ditemukan: {path}")
    manifest = {key: Path(path).resolve().relative_to(output_dir).as_posix() if path else None
                for key, path in artifacts.items()}
    manifest.update(regularization=regularization_type, refinement_iterations=refinement_iterations,
                    decimation_target=n_vertices, gaussians_per_triangle=gaussians_per_triangle,
                    white_background=white_background, scene_path=scene_path)
    manifest.update(image_downscale=image_downscale, image_max_size=image_max_size)
    manifest['mesh_export'] = 'obj' if export_obj else 'surface'
    manifest['statue_focus'] = bool(statue_focus)
    if quality_options:
        manifest['quality_options'] = quality_options
    if logger:
        manifest['stage_timings_seconds'] = dict(logger.data['timings_seconds'])
    manifest['upstream_reference'] = research_presets()['source']
    manifest['baseline_signature'] = file_signature(Path(checkpoint_3dgs) / 'point_cloud/iteration_7000/point_cloud.ply')
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)

    if logger:
        logger.log("Seluruh tahapan SuGaR berhasil diselesaikan!")


def main():
    parser = argparse.ArgumentParser(description="Pipeline SuGaR untuk satu model dataset.")
    parser.add_argument("--scene_path", "-s", type=str, required=True, help="Folder dataset (misal: datasets/Dewa/model_01).")
    parser.add_argument("--checkpoint_3dgs", "-c", type=str, required=True, help="Folder output 3DGS yang memuat checkpoint 7000.")
    parser.add_argument('--no_resume', action='store_true', help='Mulai tahap SuGaR baru tanpa memulihkan checkpoint.')
    parser.add_argument('--experiment', choices=['baseline', 'scaffold_anchor'], default='baseline')
    parser.add_argument("--output_dir", "-o", type=str, default=None, help="Folder tujuan (default: outputs/sugar/<Kelas>/<Model_ID>).")
    parser.add_argument("--statue_class", type=str, default="Dewa", help="Nama kelas patung.")
    parser.add_argument("--model_id", type=str, default="model_01", help="ID model patung.")
    parser.add_argument("--regularization", "-r", type=str, default="dn_consistency", choices=["dn_consistency", "density", "sdf"], help="Tipe regularisasi (default: dn_consistency).")
    parser.add_argument("--refinement_time", type=str, default="short", choices=["short", "medium", "long"], help="Waktu refinement (default: short / 2000 iterasi agar cepat dan aman VRAM).")
    parser.add_argument("--vertices", type=int, default=250000, help="Parameter n_vertices_in_mesh upstream, diteruskan sebagai decimation_target ke extractor.")
    parser.add_argument('--gaussians_per_triangle', type=int, choices=(1, 3, 4, 6), default=1)
    parser.add_argument('--mesh_export', choices=('obj', 'surface'), default='obj',
                        help='surface mengekspor geometri refined tanpa baking tekstur OBJ.')
    parser.add_argument('--object_region_json', default='{"mode":"camera_focus_v1"}')
    parser.add_argument('--statue_focus', action=argparse.BooleanOptionalAction, default=True,
                        help='Mesh hanya patung (tanah/objek tetangga masuk mesh background); render tetap seluruh scene.')
    parser.add_argument("--gpu", type=int, default=0, help="ID GPU device (default: 0).")
    parser.add_argument("--white_background", "-w", action="store_true", help="Gunakan background putih.")
    add_resolution_arguments(parser)

    args = parser.parse_args()
    resolution_settings(vars(args))
    sugar_settings(vars(args))
    if args.gpu < 0 or args.vertices < 1000:
        parser.error("GPU harus >= 0 dan target mesh >= 1000.")

    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "sugar" / args.statue_class / args.model_id
        if args.experiment != 'baseline':
            output_dir = PROJECT_ROOT / 'outputs/ablations' / args.experiment / 'sugar' / args.statue_class / args.model_id
    else:
        output_dir = Path(args.output_dir)

    logger = ThesisLogger(statue_class=args.statue_class, model_id=args.model_id,
                          method="SuGaR" if args.experiment == 'baseline' else 'SuGaR_scaffold_anchor')
    logger.set_parameters({
        "regularization": args.regularization,
        "refinement_time": args.refinement_time,
        "decimation_target": args.vertices,
        'gaussians_per_triangle': args.gaussians_per_triangle,
        "gpu": args.gpu,
        "phase": "training",
        'image_downscale': args.image_downscale, 'image_max_size': args.image_max_size
    })
    logger.start_vram_monitor(os.getpid())

    try:
        with gpu_lock(args.gpu):
            train_sugar_pipeline(
                scene_path=args.scene_path,
                checkpoint_3dgs=args.checkpoint_3dgs,
                resume=not args.no_resume,
                output_base_dir=str(output_dir),
                regularization_type=args.regularization,
                refinement_time=args.refinement_time,
                n_vertices=args.vertices,
                gaussians_per_triangle=args.gaussians_per_triangle,
                gpu=args.gpu,
                white_background=args.white_background,
                logger=logger,
                image_downscale=args.image_downscale,
                image_max_size=args.image_max_size,
                export_obj=args.mesh_export == 'obj',
                statue_focus=args.statue_focus
            )
            from pipeline_tools.object_region import resolve_region
            from pipeline_tools.object_mesh import export_optional
            region = resolve_region(args.scene_path, args.object_region_json)
            export_optional(PROJECT_ROOT, args.statue_class, args.model_id, 'sugar', args.experiment, region, output_dir)
        logger.stop_vram_monitor()
        logger.finalize(success=True, export_summary=not logger.data['parameters'].get('training_reused', False))
    except Exception as e:
        logger.log(f"Error pada pipeline SuGaR: {str(e)}", level="error")
        logger.stop_vram_monitor()
        logger.finalize(success=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
