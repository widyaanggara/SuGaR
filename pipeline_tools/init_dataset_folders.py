import os
from pathlib import Path

CLASSES = [
    "Dewa",
    "Dewi",
    "Mitologi",
    "Penabuh",
    "Pengapit",
    "Punakawan",
    "Raksasa",
    "Wanara"
]

MODELS = ["model_01", "model_02"]
SUBDIRS = ["images", "raw_video", "ground_truth"]

def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "datasets"
    outputs_dir = base_dir / "outputs"

    print(f"Menginisialisasi struktur folder di: {base_dir}")

    # Buat folder datasets
    for cls in CLASSES:
        for mdl in MODELS:
            for sub in SUBDIRS:
                folder_path = dataset_dir / cls / mdl / sub
                folder_path.mkdir(parents=True, exist_ok=True)
                gitkeep = folder_path / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.touch()

    # Buat folder outputs
    for sub in ["3dgs", "sugar", "logs/training_logs", "logs/evaluation_reports"]:
        folder_path = outputs_dir / sub
        folder_path.mkdir(parents=True, exist_ok=True)
        gitkeep = folder_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("Inisialisasi 8 kelas patung dan folder outputs selesai dengan sukses!")

if __name__ == "__main__":
    main()
