"""Sequential runner: each stage exits before the next one starts."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_tools.dataset_registry import (CATALOG, has_checkpoint, has_colmap_model,
                                           has_data, has_sugar_result, media_files,
                                           scene_pairs, VIDEO_EXTENSIONS, has_evaluation)
from pipeline_tools.run_settings import (add_resolution_arguments, resolution_settings,
    colmap_matches, training_matches, evaluation_matches, colmap_settings,
    add_colmap_arguments, colmap_arguments, sugar_settings, sugar_arguments)


def run_stage(script, arguments, root=PROJECT_ROOT):
    command = [sys.executable, "-u", str(root / "pipeline_tools" / script), *map(str, arguments)]
    print(f"[RUN] {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=str(root), check=True)


def process_single_statue(scene_dir, statue_class, model_id, skip_colmap=False,
                         skip_3dgs=False, skip_sugar=False, refinement_time="short",
                         gpu=0, use_gpu_colmap=False, resume=True, vertices=250000,
                         skip_evaluation=False, root=PROJECT_ROOT, colmap_max_image_size=3200,
                         image_downscale=1, image_max_size=1600, colmap_device=None,
                         colmap_matcher='exhaustive', colmap_max_features=8192,
                         colmap_sequential_overlap=10, gaussians_per_triangle=1,
                         regularization='dn_consistency'):
    scene_dir, root = Path(scene_dir), Path(root)
    sugar_config = sugar_settings(dict(vertices=vertices, refinement_time=refinement_time,
        gaussians_per_triangle=gaussians_per_triangle, regularization=regularization))
    sfm_settings = colmap_settings(dict(colmap_device=colmap_device or ('cuda' if use_gpu_colmap else 'cpu'),
        colmap_matcher=colmap_matcher, colmap_max_features=colmap_max_features,
        colmap_sequential_overlap=colmap_sequential_overlap))
    if not has_data(scene_dir):
        print(f"[SKIP] {statue_class}/{model_id}: belum ada foto/video.", flush=True)
        return "SKIPPED"
    common = ["--statue_class", statue_class, "--model_id", model_id]
    out_3dgs = root / "outputs" / "3dgs" / statue_class / model_id
    out_sugar = root / "outputs" / "sugar" / statue_class / model_id
    image_args = ['--image_downscale', image_downscale, '--image_max_size', image_max_size]
    if not media_files(scene_dir / "input") and not media_files(scene_dir / "images"):
        video = media_files(scene_dir / "raw_video", VIDEO_EXTENSIONS)[0]
        run_stage("01_extract_frames.py", ["-v", video, "-o", scene_dir / "input", *common], root)
        if not media_files(scene_dir / "input"):
            raise RuntimeError("Tidak ada frame yang lolos filter ketajaman.")
    if skip_colmap:
        if not colmap_matches(scene_dir, colmap_max_image_size, sfm_settings):
            raise RuntimeError("--skip_colmap aktif tetapi model COLMAP belum lengkap.")
    elif not (resume and colmap_matches(scene_dir, colmap_max_image_size, sfm_settings)):
        run_stage("02_run_colmap.py", ["-s", scene_dir, *common, '--gpu', gpu,
                  '--colmap_max_image_size', colmap_max_image_size,
                  *colmap_arguments(sfm_settings)], root)
    if not has_colmap_model(scene_dir):
        raise RuntimeError("Model COLMAP atau images/ hasil undistorsi tidak lengkap.")
    if not (scene_dir / 'colmap_settings.json').is_file():
        print('[INFO] Model COLMAP lama/impor dipakai kembali; resolusi SfM-nya belum tercatat. '
              'Nonaktifkan Lewati tahap jika ingin menghitung ulang dengan batas COLMAP yang dipilih.', flush=True)
    trained_3dgs = False
    if skip_3dgs:
        if not (has_checkpoint(out_3dgs, 7000) and has_checkpoint(out_3dgs, 30000)
                and training_matches(out_3dgs, scene_dir, image_downscale, image_max_size)):
            raise RuntimeError("--skip_3dgs memerlukan checkpoint 7000 dan 30000 yang lengkap.")
    elif not (resume and has_checkpoint(out_3dgs, 7000) and has_checkpoint(out_3dgs, 30000)
              and training_matches(out_3dgs, scene_dir, image_downscale, image_max_size)):
        run_stage("03_train_3dgs.py", ["-s", scene_dir, *common, "--gpu", gpu, *image_args], root)
        trained_3dgs = True
    trained_sugar = False
    sugar_matches = False
    if has_sugar_result(out_sugar):
        config = json.loads((out_sugar / 'pipeline_result.json').read_text(encoding='utf-8'))
        sugar_matches = (config.get('decimation_target', config.get('target_triangles')) == vertices and config.get('regularization') == regularization
                         and config.get('gaussians_per_triangle') == gaussians_per_triangle
                         and config.get('refinement_iterations') == {'short': 2000, 'medium': 7000, 'long': 15000}[refinement_time]
                         and config.get('image_downscale') == image_downscale and config.get('image_max_size') == image_max_size)
    if not skip_sugar and not (resume and not trained_3dgs and sugar_matches):
        run_stage("04_train_sugar.py", ["-s", scene_dir, "-c", out_3dgs, *common,
                  "--gpu", gpu, *sugar_arguments(sugar_config), *image_args], root)
        trained_sugar = True
    if not skip_evaluation:
        for method, output, retrained in (("3DGS", out_3dgs, trained_3dgs), ("SuGaR", out_sugar, trained_sugar)):
            if method == "SuGaR" and skip_sugar:
                continue
            if not (resume and not retrained and has_evaluation(output)
                    and evaluation_matches(output, image_downscale, image_max_size)):
                run_stage("06_evaluate_rendering.py", [*common, "--method", method, "--gpu", gpu, *image_args], root)
    return "COMPLETED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_name", "-c", choices=list(CATALOG))
    parser.add_argument("--model_id", "-m")
    for flag in ("skip_colmap", "skip_3dgs", "skip_sugar", "skip_evaluation", "no_resume", "use_gpu_colmap"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--refinement_time", choices=["short", "medium", "long"], default="short")
    parser.add_argument("--vertices", type=int, default=250000)
    parser.add_argument('--gaussians_per_triangle', type=int, choices=(1, 3, 4, 6), default=1)
    parser.add_argument('--regularization', choices=('dn_consistency', 'density', 'sdf'), default='dn_consistency')
    parser.add_argument("--gpu", type=int, default=0)
    add_resolution_arguments(parser, colmap=True)
    add_colmap_arguments(parser)
    args = parser.parse_args()
    resolution_settings(vars(args))
    sugar_settings(vars(args))
    if args.gpu < 0 or args.vertices < 1000:
        parser.error("GPU harus >= 0 dan target mesh >= 1000.")
    try:
        pairs = scene_pairs(args.class_name or "", args.model_id or "")
    except ValueError as exc:
        parser.error(str(exc))
    results = []
    for cls, mdl in pairs:
        print(f"\n[MODEL] {cls}/{mdl}", flush=True)
        try:
            status = process_single_statue(PROJECT_ROOT / "datasets" / cls / mdl, cls, mdl,
                skip_colmap=args.skip_colmap, skip_3dgs=args.skip_3dgs, skip_sugar=args.skip_sugar,
                refinement_time=args.refinement_time, gpu=args.gpu, use_gpu_colmap=args.use_gpu_colmap,
                resume=not args.no_resume, vertices=args.vertices, skip_evaluation=args.skip_evaluation,
                colmap_max_image_size=args.colmap_max_image_size,
                image_downscale=args.image_downscale, image_max_size=args.image_max_size,
                colmap_device=args.colmap_device, colmap_matcher=args.colmap_matcher,
                colmap_max_features=args.colmap_max_features,
                colmap_sequential_overlap=args.colmap_sequential_overlap,
                gaussians_per_triangle=args.gaussians_per_triangle, regularization=args.regularization)
            results.append({"target": f"{cls}/{mdl}", "status": status})
        except Exception as exc:
            print(f"[ERROR] {cls}/{mdl}: {exc}", flush=True)
            results.append({"target": f"{cls}/{mdl}", "status": "FAILED", "error": str(exc)})
    print("[BATCH SUMMARY] " + json.dumps(results), flush=True)
    return 1 if any(item["status"] == "FAILED" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
