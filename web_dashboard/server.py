import os
import sys
import json
import mimetypes
import time
import subprocess
import threading
import csv
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

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

# Global job manager
CURRENT_JOB = {
    "is_running": False,
    "action": None,
    "target": None,
    "process": None,
    "logs": [],
    "start_time": None,
    "status": "IDLE"  # IDLE, RUNNING, COMPLETED, ERROR, STOPPED
}

JOB_LOCK = threading.Lock()


def get_project_status():
    """Memindai folder datasets/ dan outputs/ untuk mengetahui status tiap model."""
    dataset_dir = PROJECT_ROOT / "datasets"
    output_dir = PROJECT_ROOT / "outputs"

    classes_data = {}

    for cls in CLASSES:
        models_data = {}
        for mdl in MODELS:
            model_scene = dataset_dir / cls / mdl
            images_dir = model_scene / "images"
            video_dir = model_scene / "raw_video"
            sparse_dir = model_scene / "sparse" / "0"
            gt_dir = model_scene / "ground_truth"

            out_3dgs = output_dir / "3dgs" / cls / mdl
            out_sugar = output_dir / "sugar" / cls / mdl

            # Hitung foto
            img_count = len([f for f in images_dir.iterdir() if f.is_file()]) if images_dir.exists() else 0
            has_video = len(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MP4"))) > 0 if video_dir.exists() else False
            has_gt = len(list(gt_dir.glob("*.ply"))) > 0 if gt_dir.exists() else False

            # Status COLMAP
            colmap_done = sparse_dir.exists() and (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists()

            # Status 3DGS
            dgs_7k = (out_3dgs / "point_cloud" / "iteration_7000").exists() if out_3dgs.exists() else False
            dgs_30k = (out_3dgs / "point_cloud" / "iteration_30000").exists() if out_3dgs.exists() else False

            # Status SuGaR
            sugar_mesh = (out_sugar / "mesh").exists() and len(list((out_sugar / "mesh").glob("*.ply"))) > 0 if out_sugar.exists() else False
            sugar_obj = (out_sugar / "textured_mesh").exists() and len(list((out_sugar / "textured_mesh").glob("*.obj"))) > 0 if out_sugar.exists() else False

            models_data[mdl] = {
                "images_count": img_count,
                "has_video": has_video,
                "has_gt_lidar": has_gt,
                "colmap_done": colmap_done,
                "3dgs_7k": dgs_7k,
                "3dgs_30k": dgs_30k,
                "sugar_mesh": sugar_mesh,
                "sugar_obj": sugar_obj,
                "overall_status": "Completed" if (dgs_30k and sugar_obj) else ("Ready" if img_count > 0 else "No Photos")
            }
        classes_data[cls] = models_data

    return classes_data


def get_metrics_table():
    """Membaca outputs/logs/evaluation_reports/summary_comparison.csv."""
    csv_file = PROJECT_ROOT / "outputs" / "logs" / "evaluation_reports" / "summary_comparison.csv"
    if not csv_file.exists():
        return []

    rows = []
    try:
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        print(f"Error reading metrics CSV: {e}")
    return rows


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_preview_images(statue_class: str, model_id: str):
    """Memindai folder outputs/ untuk mencari gambar hasil render nyata tiap model."""
    output_dir = PROJECT_ROOT / "outputs"
    items = []

    # 1. 3DGS renders (test evaluasi / train renders)
    dirs_to_check_3dgs = [
        output_dir / "3dgs" / statue_class / model_id / "test" / "ours_30000" / "renders",
        output_dir / "3dgs" / statue_class / model_id / "train" / "ours_30000" / "renders",
        output_dir / "3dgs" / statue_class / model_id / "test" / "ours_7000" / "renders",
        output_dir / "3dgs" / statue_class / model_id / "renders",
    ]
    for rdir in dirs_to_check_3dgs:
        if rdir.exists():
            imgs = sorted([f for f in rdir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
            if imgs:
                rel = rdir.relative_to(PROJECT_ROOT).as_posix()
                for i, img in enumerate(imgs[:3]):
                    items.append({
                        "src": f"/static/{rel}/{img.name}",
                        "label": f"3DGS #{i+1}",
                        "type": "3dgs"
                    })
                break

    # 2. SuGaR mesh preview
    sugar_dirs = [
        output_dir / "sugar" / statue_class / model_id / "textured_mesh",
        output_dir / "sugar" / statue_class / model_id / "renders",
        output_dir / "sugar" / statue_class / model_id / "evaluation",
    ]
    for sdir in sugar_dirs:
        if sdir.exists():
            imgs = sorted([f for f in sdir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
            if imgs:
                rel = sdir.relative_to(PROJECT_ROOT).as_posix()
                for i, img in enumerate(imgs[:3]):
                    items.append({
                        "src": f"/static/{rel}/{img.name}",
                        "label": f"SuGaR #{i+1}",
                        "type": "sugar"
                    })
                break

    return {
        "images": items
    }


def run_pipeline_task(action: str, statue_class: str, model_id: str, gpu: int = 0):
    """Menjalankan proses di background thread."""
    global CURRENT_JOB

    with JOB_LOCK:
        CURRENT_JOB["is_running"] = True
        CURRENT_JOB["action"] = action
        CURRENT_JOB["target"] = f"{statue_class}/{model_id}" if statue_class and model_id else "ALL_MODELS"
        CURRENT_JOB["logs"] = []
        CURRENT_JOB["start_time"] = time.time()
        CURRENT_JOB["status"] = "RUNNING"

    scene_path = PROJECT_ROOT / "datasets" / statue_class / model_id if statue_class and model_id else None

    # Tentukan command
    cmd = []
    if action == "full_pipeline":
        script = PROJECT_ROOT / "pipeline_tools" / "batch_run_all.py"
        cmd = [sys.executable, str(script), "--gpu", str(gpu)]
        if statue_class:
            cmd.extend(["--class_name", statue_class])
        if model_id:
            cmd.extend(["--model_id", model_id])

    elif action == "colmap":
        script = PROJECT_ROOT / "pipeline_tools" / "02_run_colmap.py"
        cmd = [sys.executable, str(script), "-s", str(scene_path), "--statue_class", statue_class, "--model_id", model_id]

    elif action == "3dgs":
        script = PROJECT_ROOT / "pipeline_tools" / "03_train_3dgs.py"
        cmd = [sys.executable, str(script), "-s", str(scene_path), "--statue_class", statue_class, "--model_id", model_id, "--gpu", str(gpu)]

    elif action == "sugar":
        script = PROJECT_ROOT / "pipeline_tools" / "04_train_sugar.py"
        out_3dgs = PROJECT_ROOT / "outputs" / "3dgs" / statue_class / model_id
        cmd = [sys.executable, str(script), "-s", str(scene_path), "-c", str(out_3dgs), "--statue_class", statue_class, "--model_id", model_id, "--refinement_time", "short", "--gpu", str(gpu)]

    elif action == "extract_frames":
        script = PROJECT_ROOT / "pipeline_tools" / "01_extract_frames.py"
        raw_vid_dir = scene_path / "raw_video"
        vids = list(raw_vid_dir.glob("*.mp4")) + list(raw_vid_dir.glob("*.MP4"))
        if vids:
            cmd = [sys.executable, str(script), "-v", str(vids[0]), "-o", str(scene_path / "images"), "--fps", "2.0", "--statue_class", statue_class, "--model_id", model_id]
        else:
            with JOB_LOCK:
                CURRENT_JOB["logs"].append(f"[ERROR] Tidak ada file video di {raw_vid_dir}")
                CURRENT_JOB["is_running"] = False
                CURRENT_JOB["status"] = "ERROR"
            return

    elif action == "evaluate":
        script = PROJECT_ROOT / "pipeline_tools" / "05_evaluate_comparison.py"
        sugar_obj = PROJECT_ROOT / "outputs" / "sugar" / statue_class / model_id / "textured_mesh" / "model.obj"
        gt_lidar = scene_path / "ground_truth" / "lidar.ply"
        cmd = [sys.executable, str(script), "-c", statue_class, "-m", model_id, "--method", "SuGaR", "-p", str(sugar_obj)]
        if gt_lidar.exists():
            cmd.extend(["-g", str(gt_lidar)])

    with JOB_LOCK:
        CURRENT_JOB["logs"].append(f"[WEB RUNNER] Menjalankan perintah: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        with JOB_LOCK:
            CURRENT_JOB["process"] = proc

        for line in proc.stdout:
            clean_line = line.rstrip()
            if clean_line:
                with JOB_LOCK:
                    CURRENT_JOB["logs"].append(clean_line)
                    # Batasi memory log buffer ke 2000 baris terakhir
                    if len(CURRENT_JOB["logs"]) > 2000:
                        CURRENT_JOB["logs"] = CURRENT_JOB["logs"][-2000:]

        proc.wait()

        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["process"] = None
            if proc.returncode == 0:
                CURRENT_JOB["status"] = "COMPLETED"
                CURRENT_JOB["logs"].append("\n[WEB RUNNER] >>> PROSES BERHASIL DISELESAIKAN DENGAN SUKSES! <<<")
            else:
                CURRENT_JOB["status"] = "ERROR"
                CURRENT_JOB["logs"].append(f"\n[WEB RUNNER] >>> PROSES GAGAL DENGAN EXIT CODE {proc.returncode} <<<")

    except Exception as e:
        with JOB_LOCK:
            CURRENT_JOB["is_running"] = False
            CURRENT_JOB["process"] = None
            CURRENT_JOB["status"] = "ERROR"
            CURRENT_JOB["logs"].append(f"\n[WEB RUNNER EXCEPTION] {str(e)}")


class DashboardHandler(BaseHTTPRequestHandler):
    """Handler HTTP server lokal untuk web dashboard."""

    def do_GET(self):
        url = urlparse(self.path)

        if url.path == "/" or url.path == "/index.html":
            self.serve_index()
        elif url.path == "/api/status":
            self.send_json(get_project_status())
        elif url.path == "/api/metrics":
            self.send_json(get_metrics_table())
        elif url.path.startswith("/api/previews/"):
            parts = url.path.strip("/").split("/")
            if len(parts) >= 4:
                self.send_json(get_preview_images(parts[2], parts[3]))
            else:
                self.send_json({"error": "Format: /api/previews/<class>/<model>"})
        elif url.path == "/api/logs":
            with JOB_LOCK:
                elapsed = time.time() - CURRENT_JOB["start_time"] if CURRENT_JOB["start_time"] else 0
                resp = {
                    "is_running": CURRENT_JOB["is_running"],
                    "action": CURRENT_JOB["action"],
                    "target": CURRENT_JOB["target"],
                    "status": CURRENT_JOB["status"],
                    "elapsed_seconds": round(elapsed, 1),
                    "logs": CURRENT_JOB["logs"][-300:]  # 300 baris terbaru
                }
            self.send_json(resp)
        elif url.path.startswith("/static/"):
            self.serve_static(url.path)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        url = urlparse(self.path)

        if url.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            with JOB_LOCK:
                if CURRENT_JOB["is_running"]:
                    self.send_json({"error": "Proses lain sedang berjalan. Harap tunggu atau klik Stop."}, status=400)
                    return

            action = data.get("action", "full_pipeline")
            statue_class = data.get("class_name", "")
            model_id = data.get("model_id", "")
            gpu = int(data.get("gpu", 0))

            # Mulai thread di background
            t = threading.Thread(
                target=run_pipeline_task,
                args=(action, statue_class, model_id, gpu),
                daemon=True
            )
            t.start()

            self.send_json({"message": f"Tugas '{action}' untuk {statue_class}/{model_id} berhasil dimulai!"})

        elif url.path == "/api/stop":
            with JOB_LOCK:
                if CURRENT_JOB["process"] and CURRENT_JOB["is_running"]:
                    try:
                        CURRENT_JOB["process"].terminate()
                        CURRENT_JOB["is_running"] = False
                        CURRENT_JOB["status"] = "STOPPED"
                        CURRENT_JOB["logs"].append("\n[WEB RUNNER] >>> PROSES DIHENTIKAN OLEH PENGGUNA <<<")
                        self.send_json({"message": "Proses berhasil dihentikan!"})
                    except Exception as e:
                        self.send_json({"error": str(e)}, status=500)
                else:
                    self.send_json({"message": "Tidak ada proses yang sedang berjalan."})
        else:
            self.send_error(404, "Not Found")

    def serve_index(self):
        html_file = Path(__file__).resolve().parent / "index.html"
        if not html_file.exists():
            self.send_error(500, "index.html tidak ditemukan")
            return

        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def serve_static(self, url_path: str):
        """Serve file statis dari folder outputs/."""
        # Hapus prefix /static/
        rel = url_path[len("/static/"):]

        if not rel.startswith("outputs/"):
            self.send_error(404, "Not Found")
            return

        file_path = (PROJECT_ROOT / rel).resolve()
        outputs_root = (PROJECT_ROOT / "outputs").resolve()
        if not str(file_path).lower().startswith(str(outputs_root).lower()):
            self.send_error(403, "Forbidden")
            return

        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File Not Found")
            return

        # Tentukan MIME type
        mime, _ = mimetypes.guess_type(str(file_path))
        if mime is None:
            mime = "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=300")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_error(500, "Internal Server Error")

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def log_message(self, format, *args):
        # Meredam pesan log HTTP request agar tidak memenuhi terminal
        pass


def run_server(port=8080):
    server_address = ("", port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print(f"===========================================================")
    print(f" Web Dashboard Skripsi: 3DGS vs SuGaR (8 Kelas Patung Bali)")
    print(f" Berjalan di: http://localhost:{port}")
    print(f" Buka browser Anda untuk memonitor progress & menjalankan pipeline!")
    print(f" Tekan Ctrl+C untuk keluar.")
    print(f"===========================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nMenutup server dashboard...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port=port)
