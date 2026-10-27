"""Create the 18 dataset slots without replacing existing data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_tools.dataset_registry import PROJECT_ROOT, scene_pairs


def initialize_workspace(base_dir=PROJECT_ROOT):
    base_dir = Path(base_dir)
    for cls, mdl in scene_pairs():
        for sub in ("input", "images", "raw_video", "ground_truth"):
            folder = base_dir / "datasets" / cls / mdl / sub
            folder.mkdir(parents=True, exist_ok=True)
            (folder / ".gitkeep").touch(exist_ok=True)
        for method, subdirs in {"3dgs": ("point_cloud", "evaluation"),
                               "sugar": ("coarse", "mesh", "refined", "textured_mesh", "evaluation")}.items():
            for sub in subdirs:
                (base_dir / "outputs" / method / cls / mdl / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("logs/training_logs", "logs/evaluation_reports", "queue/logs", "locks"):
        (base_dir / "outputs" / sub).mkdir(parents=True, exist_ok=True)
    return len(scene_pairs())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    print(f"Folder input/output {initialize_workspace(args.root)} model siap di {args.root.resolve()}")


if __name__ == "__main__":
    main()
