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


def run_stage(script, arguments, root=PROJECT_ROOT):
    command = [sys.executable, "-u", str(root / "pipeline_tools" / script), *map(str, arguments)]
    print(f"[RUN] {subprocess.list2cmdline(command)}", flush=True)
    subprocess.run(command, cwd=str(root), check=True)


def process_single_statue(scene_dir, statue_class, model_id, skip_colmap=False,
                         skip_3dgs=False, skip_sugar=False, refinement_time="short",
                         gpu=0, use_gpu_colmap=False, resume=True, vertices=250000,
                         skip_evaluation=False, root=PROJECT_ROOT):
    scene_dir, root = Path(scene_dir), Path(root)
    if not has_data(scene_dir):
        print(f"[SKIP] {statue_class}/{model_id}: belum ada foto/video.", flush=True)
        return "SKIPPED"
    common = ["--statue_class", statue_class, "--model_id", model_id]
    out_3dgs = root / "outputs" / "3dgs" / statue_class / model_id
    out_sugar = root / "outputs" / "sugar" / statue_class / model_id
    if not media_files(scene_dir / "input") and not media_files(scene_dir / "images"):
        video = media_files(scene_dir / "raw_video", VIDEO_EXTENSIONS)[0]
        run_stage("01_extract_frames.py", ["-v", video, "-o", scene_dir / "input", *common], root)
        if not media_files(scene_dir / "input"):
            raise RuntimeError("Tidak ada frame yang lolos filter ketajaman.")
    if skip_colmap:
        if not has_colmap_model(scene_dir):
            raise RuntimeError("--skip_colmap aktif tetapi model COLMAP belum lengkap.")
    elif not (resume and has_colmap_model(scene_dir)):
        run_stage("02_run_colmap.py", ["-s", scene_dir, *common, '--gpu', gpu,
                  *(["--use_gpu"] if use_gpu_colmap else [])], root)
    if not has_colmap_model(scene_dir):
        raise RuntimeError("Model COLMAP atau images/ hasil undistorsi tidak lengkap.")
    trained_3dgs = False
    if skip_3dgs:
        if not has_checkpoint(out_3dgs, 7000) or not has_checkpoint(out_3dgs, 30000):
            raise RuntimeError("--skip_3dgs memerlukan checkpoint 7000 dan 30000 yang lengkap.")
    elif not (resume and has_checkpoint(out_3dgs, 7000) and has_checkpoint(out_3dgs, 30000)):
        run_stage("03_train_3dgs.py", ["-s", scene_dir, *common, "--gpu", gpu], root)
        trained_3dgs = True
    trained_sugar = False
    sugar_matches = False
    if has_sugar_result(out_sugar):
        config = json.loads((out_sugar / 'pipeline_result.json').read_text(encoding='utf-8'))
        sugar_matches = (config.get('target_triangles') == vertices and config.get('regularization') == 'dn_consistency'
                         and config.get('refinement_iterations') == {'short': 2000, 'medium': 7000, 'long': 15000}[refinement_time])
    if not skip_sugar and not (resume and not trained_3dgs and sugar_matches):
        run_stage("04_train_sugar.py", ["-s", scene_dir, "-c", out_3dgs, *common,
                  "--gpu", gpu, "--refinement_time", refinement_time, "--vertices", vertices], root)
        trained_sugar = True
    if not skip_evaluation:
        for method, output, retrained in (("3DGS", out_3dgs, trained_3dgs), ("SuGaR", out_sugar, trained_sugar)):
            if method == "SuGaR" and skip_sugar:
                continue
            if not (resume and not retrained and has_evaluation(output)):
                run_stage("06_evaluate_rendering.py", [*common, "--method", method, "--gpu", gpu], root)
    return "COMPLETED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class_name", "-c", choices=list(CATALOG))
    parser.add_argument("--model_id", "-m")
    for flag in ("skip_colmap", "skip_3dgs", "skip_sugar", "skip_evaluation", "no_resume", "use_gpu_colmap"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--refinement_time", choices=["short", "medium", "long"], default="short")
    parser.add_argument("--vertices", type=int, default=250000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
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
                resume=not args.no_resume, vertices=args.vertices, skip_evaluation=args.skip_evaluation)
            results.append({"target": f"{cls}/{mdl}", "status": status})
        except Exception as exc:
            print(f"[ERROR] {cls}/{mdl}: {exc}", flush=True)
            results.append({"target": f"{cls}/{mdl}", "status": "FAILED", "error": str(exc)})
    print("[BATCH SUMMARY] " + json.dumps(results), flush=True)
    return 1 if any(item["status"] == "FAILED" for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
