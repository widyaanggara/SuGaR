import os
import sys
import time
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ThesisLogger:
    """
    Sistem pencatatan log otomatis untuk eksperimen Skripsi:
    Komparasi 3D Gaussian Splatting (3DGS) vs SuGaR.
    
    Menghasilkan:
    1. Log teks detail per run: outputs/logs/training_logs/<session_name>.log
    2. Data terstruktur JSON per run: outputs/logs/training_logs/<session_name>.json
    3. Tabel rekapitulasi CSV (Excel-ready): outputs/logs/evaluation_reports/summary_comparison.csv
    4. Tabel rekapitulasi Markdown (Thesis-ready): outputs/logs/evaluation_reports/summary_comparison.md
    """

    def __init__(self, 
                 statue_class: str, 
                 model_id: str, 
                 method: str = "SuGaR",
                 base_dir: Optional[str] = None):
        self.statue_class = statue_class
        self.model_id = model_id
        self.method = method  # "3DGS" atau "SuGaR"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{statue_class}_{model_id}_{method}_{self.timestamp}"

        # Setup direktori
        if base_dir is None:
            # Mengacu ke root repository SuGaR
            self.base_dir = Path(__file__).resolve().parent.parent
        else:
            self.base_dir = Path(base_dir).resolve()

        self.log_dir = self.base_dir / "outputs" / "logs" / "training_logs"
        self.report_dir = self.base_dir / "outputs" / "logs" / "evaluation_reports"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{self.session_name}.log"
        self.json_file = self.log_dir / f"{self.session_name}.json"
        self.csv_file = self.report_dir / "summary_comparison.csv"
        self.md_file = self.report_dir / "summary_comparison.md"

        # Setup logger
        self._init_logger()

        # Data metrik penampung
        self.data: Dict[str, Any] = {
            "session_name": self.session_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statue_class": self.statue_class,
            "model_id": self.model_id,
            "method": self.method,
            "timings_seconds": {},
            "peak_vram_gb": 0.0,
            "metrics": {
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "chamfer_distance_cm": None,
                "num_elements": None  # Gaussians untuk 3DGS, Vertices untuk SuGaR
            },
            "parameters": {},
            "status": "IN_PROGRESS"
        }

        self._timers: Dict[str, float] = {}

    def _init_logger(self):
        self.logger = logging.getLogger(self.session_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Handler ke file
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Handler ke konsol terminal
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def log(self, message: str, level: str = "info"):
        """Mencatat pesan ke log file dan terminal."""
        getattr(self.logger, level.lower(), self.logger.info)(message)

    def start_stage(self, stage_name: str):
        """Memulai pencatatan durasi sebuah tahapan."""
        self._timers[stage_name] = time.time()
        self.log(f">>> [START STAGE] {stage_name}")

    def end_stage(self, stage_name: str) -> float:
        """Mengakhiri pencatatan durasi dan mencatat ke memori."""
        if stage_name not in self._timers:
            self.log(f"Peringatan: Timer '{stage_name}' belum pernah dimulai!", level="warning")
            return 0.0

        elapsed = time.time() - self._timers.pop(stage_name)
        elapsed_min = elapsed / 60.0
        self.data["timings_seconds"][stage_name] = round(elapsed, 2)
        self.log(f"<<< [END STAGE] {stage_name} diselesaikan dalam {elapsed:.2f} detik ({elapsed_min:.2f} menit)")
        self.record_vram()
        return elapsed

    def record_vram(self):
        """Mencatat penggunaan memori VRAM GPU tertinggi."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_gb = round(peak_bytes / (1024 ** 3), 3)
            if peak_gb > self.data["peak_vram_gb"]:
                self.data["peak_vram_gb"] = peak_gb
                self.log(f"Puncak Penggunaan VRAM GPU: {peak_gb:.2f} GB")

    def set_metrics(self, 
                    psnr: Optional[float] = None, 
                    ssim: Optional[float] = None, 
                    lpips: Optional[float] = None,
                    chamfer_distance_cm: Optional[float] = None,
                    num_elements: Optional[int] = None):
        """Merekam hasil evaluasi metrik kuantitatif."""
        if psnr is not None:
            self.data["metrics"]["psnr"] = round(float(psnr), 4)
        if ssim is not None:
            self.data["metrics"]["ssim"] = round(float(ssim), 4)
        if lpips is not None:
            self.data["metrics"]["lpips"] = round(float(lpips), 4)
        if chamfer_distance_cm is not None:
            self.data["metrics"]["chamfer_distance_cm"] = round(float(chamfer_distance_cm), 4)
        if num_elements is not None:
            self.data["metrics"]["num_elements"] = int(num_elements)

        self.log(f"[METRICS UPDATED] PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, Chamfer: {chamfer_distance_cm} cm")

    def set_parameters(self, params_dict: Dict[str, Any]):
        """Menyimpan konfigurasi parameter eksperimen."""
        self.data["parameters"].update(params_dict)

    def finalize(self, success: bool = True):
        """Menyelesaikan sesi log dan mengekspor rekapitulasi ke JSON, CSV, dan Markdown."""
        self.data["status"] = "COMPLETED" if success else "FAILED"
        self.record_vram()

        # Hitung total waktu
        total_seconds = sum(self.data["timings_seconds"].values())
        self.data["total_duration_seconds"] = round(total_seconds, 2)
        self.data["total_duration_minutes"] = round(total_seconds / 60.0, 2)

        self.log(f"==================================================")
        self.log(f"SESI SELESAI: {self.session_name}")
        self.log(f"Status: {self.data['status']}")
        self.log(f"Total Waktu Komputasi: {self.data['total_duration_minutes']} menit")
        self.log(f"Puncak VRAM: {self.data['peak_vram_gb']} GB")
        self.log(f"==================================================")

        # 1. Simpan JSON terstruktur
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

        # 2. Update CSV Rekapitulasi Master
        self._append_to_csv()

        # 3. Regenerate Markdown Report
        self._generate_markdown_report()

    def _append_to_csv(self):
        """Menulis atau memperbarui baris data di file master CSV."""
        headers = [
            "Session Name", "Timestamp", "Kelas", "Model ID", "Metode",
            "Total Waktu (Menit)", "Peak VRAM (GB)", 
            "PSNR (dB)", "SSIM", "LPIPS", "Chamfer Dist (cm)", 
            "Jumlah Titik/Vertex", "Status"
        ]

        row = [
            self.session_name,
            self.data["timestamp"],
            self.statue_class,
            self.model_id,
            self.method,
            self.data.get("total_duration_minutes", 0),
            self.data["peak_vram_gb"],
            self.data["metrics"]["psnr"] if self.data["metrics"]["psnr"] is not None else "-",
            self.data["metrics"]["ssim"] if self.data["metrics"]["ssim"] is not None else "-",
            self.data["metrics"]["lpips"] if self.data["metrics"]["lpips"] is not None else "-",
            self.data["metrics"]["chamfer_distance_cm"] if self.data["metrics"]["chamfer_distance_cm"] is not None else "-",
            self.data["metrics"]["num_elements"] if self.data["metrics"]["num_elements"] is not None else "-",
            self.data["status"]
        ]

        file_exists = self.csv_file.exists()
        with open(self.csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row)

    def _generate_markdown_report(self):
        """Membaca master CSV dan membuat tabel Markdown rapi yang siap disalin ke Bab 4 Skripsi."""
        if not self.csv_file.exists():
            return

        with open(self.csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return

        headers = rows[0]
        data_rows = rows[1:]

        md_content = []
        md_content.append("# Tabel Rekapitulasi Hasil Komparasi 3DGS vs SuGaR")
        md_content.append("")
        md_content.append(f"> Terakhir diperbarui: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md_content.append("")
        md_content.append("Tabel di bawah ini merangkum metrik performa komputasi dan kualitas rekonstruksi 3D yang dapat langsung disalin ke **Bab 4 (Hasil dan Pembahasan) Skripsi**:")
        md_content.append("")

        # Format header markdown
        md_content.append("| " + " | ".join(headers[2:]) + " |")
        md_content.append("| " + " | ".join(["---"] * len(headers[2:])) + " |")

        for r in data_rows:
            # Ambil kolom dari indeks ke-2 (mengabaikan Session Name dan Timestamp agar tabel ringkas)
            md_content.append("| " + " | ".join(r[2:]) + " |")

        md_content.append("")
        md_content.append("### Keterangan Metrik:")
        md_content.append("- **PSNR (dB)**: Semakin tinggi semakin baik (kualitas rendering novel view).")
        md_content.append("- **SSIM**: Nilai rentang 0–1, semakin mendekati 1 semakin mirip struktur citra hasil rekonstruksi.")
        md_content.append("- **LPIPS**: Metrik persepsi manusia, semakin rendah nilainya semakin realistis tekstur rendering.")
        md_content.append("- **Chamfer Dist (cm)**: Jarak rata-rata geometri hasil rekonstruksi terhadap data acuan LiDAR (Ground Truth).")
        md_content.append("- **Peak VRAM**: Puncak penggunaan memori GPU selama proses berjalan.")

        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
