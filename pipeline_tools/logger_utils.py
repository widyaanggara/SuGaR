import os
import sys
import time
import json
import csv
import math
import logging
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from pipeline_tools.geometry_metrics import GEOMETRY_COLUMNS

# The queue/3DGS supervisor must never create its own CUDA context.
# Inspect torch only if the current process has already imported it.


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
        self._started_at = time.monotonic()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
            "vram_measurement": "unavailable",
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
        self._vram_monitor_stop = None
        self._vram_monitor_thread = None
        self._vram_monitor_pid = None

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
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_gb = round(peak_bytes / (1024 ** 3), 3)
            self.data['peak_vram_gb'] = peak_gb
            self.data["vram_measurement"] = "torch_peak_allocated"

    def record_process_vram_gb(self, used_gb: float):
        """Update VRAM peak from either PyTorch or nvidia-smi process sampling."""
        used_gb = max(0.0, float(used_gb))
        if used_gb > self.data["peak_vram_gb"]:
            self.data["peak_vram_gb"] = round(used_gb, 3)
            self.log(f"Puncak Penggunaan VRAM GPU: {used_gb:.2f} GB")

    def start_vram_monitor(self, pid: Optional[int] = None, interval_seconds: float = 1.0):
        """Sample this process (or a child training process) with nvidia-smi."""
        self.stop_vram_monitor()
        if shutil.which("nvidia-smi") is None:
            self.log("nvidia-smi tidak ditemukan; pencatatan VRAM proses tidak aktif.", level="warning")
            return False

        self._vram_monitor_pid = int(pid if pid is not None else os.getpid())
        stop_event = threading.Event()
        self._vram_monitor_stop = stop_event

        def sample_process_memory():
            command = [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
                if result.returncode != 0:
                    return
                used_mb = 0.0
                found_pid = False
                for row in result.stdout.splitlines():
                    columns = [column.strip() for column in row.split(",", 1)]
                    if len(columns) != 2 or columns[0] != str(self._vram_monitor_pid):
                        continue
                    try:
                        used_mb += float(columns[1].split()[0])
                        found_pid = True
                    except (ValueError, IndexError):
                        continue
                if found_pid:
                    self.record_process_vram_gb(used_mb / 1024.0)
                    self.data["vram_measurement"] = "nvidia_smi_process_sample"
            except (OSError, subprocess.SubprocessError):
                return

        def monitor():
            while not stop_event.is_set():
                sample_process_memory()
                stop_event.wait(max(0.2, interval_seconds))

        self._vram_monitor_thread = threading.Thread(target=monitor, name="vram-monitor", daemon=True)
        self._vram_monitor_thread.start()
        return True

    def stop_vram_monitor(self):
        stop_event = self._vram_monitor_stop
        monitor_thread = self._vram_monitor_thread
        if stop_event is None:
            return
        stop_event.set()
        if monitor_thread is not None and monitor_thread is not threading.current_thread():
            monitor_thread.join(timeout=6)
        self._vram_monitor_stop = None
        self._vram_monitor_thread = None
        self._vram_monitor_pid = None

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

    def set_geometry_metrics(self, metrics):
        """Record both geometric thresholds explicitly; missing data stays missing."""
        for key in GEOMETRY_COLUMNS:
            if key in metrics:
                value = float(metrics[key])
                if not math.isfinite(value) or not 0 <= value <= 100:
                    raise ValueError(f'{key} harus finite dalam rentang 0–100%.')
                self.data['metrics'][key] = round(value, 4)

    def finalize(self, success: bool = True):
        """Menyelesaikan sesi log dan mengekspor rekapitulasi ke JSON, CSV, dan Markdown."""
        self.data["status"] = "COMPLETED" if success else "FAILED"
        self.stop_vram_monitor()
        self.record_vram()

        # Hitung total waktu
        total_seconds = time.monotonic() - self._started_at
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
            self.data["peak_vram_gb"] if self.data["vram_measurement"] != "unavailable" else "-",
            self.data["metrics"]["psnr"] if self.data["metrics"]["psnr"] is not None else "-",
            self.data["metrics"]["ssim"] if self.data["metrics"]["ssim"] is not None else "-",
            self.data["metrics"]["lpips"] if self.data["metrics"]["lpips"] is not None else "-",
            self.data["metrics"]["chamfer_distance_cm"] if self.data["metrics"]["chamfer_distance_cm"] is not None else "-",
            self.data["metrics"]["num_elements"] if self.data["metrics"]["num_elements"] is not None else "-",
            self.data["status"]
        ]

        values = dict(zip(headers, row))
        for key, column in GEOMETRY_COLUMNS.items():
            values[column] = self.data['metrics'].get(key, '-')
        headers.extend(GEOMETRY_COLUMNS.values())
        previous = []
        if self.csv_file.is_file() and self.csv_file.stat().st_size:
            with self.csv_file.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                old_headers = reader.fieldnames or []
                previous = list(reader)
            # Preserve historical data and any additional columns during schema upgrades.
            headers.extend(column for column in old_headers if column not in headers)
            if any(None in entry for entry in previous):
                raise ValueError('CSV lama memiliki baris tidak sesuai header; tidak ditimpa.')
        temporary = self.csv_file.with_name(self.csv_file.name + f'.{os.getpid()}.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, restval='-')
            writer.writeheader()
            writer.writerows(previous)
            writer.writerow(values)
        temporary.replace(self.csv_file)

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
        md_content.append('- **Precision / Recall / F1 @2cm dan @5cm (%)**: Metrik geometri terhadap LiDAR, bukan klasifikasi kelas patung. Precision mengukur titik rekonstruksi dekat ground truth; Recall mengukur cakupan ground truth. F1 = 2PR/(P+R), atau 0 bila P+R=0. Ambang memakai jarak < 2 cm / < 5 cm.')
        md_content.append("- **Peak VRAM**: Puncak penggunaan memori GPU selama proses berjalan.")

        with open(self.md_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
