"""Shared dataset catalogue and artifact checks (no GPU dependencies)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BALI_CLASSES = ("Dewa", "Dewi", "Mitologi", "Penabuh", "Pengapit", "Punakawan", "Raksasa", "Wanara")
CATALOG = {name: ("model_01", "model_02") for name in BALI_CLASSES}
CATALOG["TanksAndTemples"] = ("tanks_and_temples_1", "tanks_and_temples_2")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def media_files(directory, extensions=IMAGE_EXTENSIONS):
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions) if directory.is_dir() else []


def scene_pairs(class_name="", model_id=""):
    if class_name and class_name not in CATALOG:
        raise ValueError("Kelas dataset tidak dikenal.")
    pairs = [(cls, mdl) for cls, models in CATALOG.items() for mdl in models
             if (not class_name or cls == class_name) and (not model_id or mdl == model_id)]
    if not pairs:
        raise ValueError("Model tidak terdaftar untuk kelas tersebut.")
    return pairs


def has_colmap_model(scene):
    sparse = Path(scene) / "sparse" / "0"
    return bool(media_files(Path(scene) / "images")) and any(
        all((sparse / (name + ext)).is_file() and (sparse / (name + ext)).stat().st_size > 0
            for name in ("cameras", "images", "points3D")) for ext in (".bin", ".txt"))


def has_checkpoint(output, iteration):
    output = Path(output)
    ply = output / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return ply.is_file() and ply.stat().st_size > 0 and (output / "cameras.json").is_file()


def file_signature(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def has_sugar_result(output):
    import json
    output = Path(output)
    try:
        manifest = json.loads((output / "pipeline_result.json").read_text(encoding="utf-8"))
        baseline = output.parents[2] / '3dgs' / output.parent.name / output.name / 'point_cloud/iteration_7000/point_cloud.ply'
        if 'baseline_signature' in manifest and manifest['baseline_signature'] != file_signature(baseline):
            return False
        return all((output / manifest[key]).is_file() and (output / manifest[key]).stat().st_size > 0
                   for key in ("coarse_mesh", "refined_checkpoint", "textured_mesh"))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def has_evaluation(output):
    import json
    output = Path(output)
    try:
        report = json.loads((output / 'evaluation/metrics.json').read_text(encoding='utf-8'))
        return (len(report['per_view']) > 0 and report['checkpoint_signature'] == file_signature(output / report['checkpoint'])
                and all(isinstance(report[key], (int, float)) for key in ('psnr', 'ssim', 'lpips')))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def has_data(scene):
    scene = Path(scene)
    return bool(media_files(scene / "input") or media_files(scene / "images") or
                media_files(scene / "raw_video", VIDEO_EXTENSIONS))
