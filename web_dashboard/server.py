"""Local dashboard and persistent, single-worker pipeline queue."""
import argparse
import csv
import json
import math
import mimetypes
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_tools.dataset_registry import (CATALOG, VIDEO_EXTENSIONS, has_checkpoint,
    has_colmap_model, has_data, has_sugar_result, has_evaluation, media_files, scene_pairs)
from pipeline_tools.init_dataset_folders import initialize_workspace
from pipeline_tools.gpu_lock import gpu_lock
from web_dashboard.job_queue import JobQueue
from pipeline_tools.run_settings import (resolution_settings, colmap_settings, colmap_arguments,
    sugar_settings, sugar_arguments, research_presets)
from pipeline_tools.colmap_capabilities import inspect_colmap


def get_project_status(root=PROJECT_ROOT):
    root = Path(root)
    result = {}
    for cls, mdl in scene_pairs():
        scene = root / "datasets" / cls / mdl
        dgs = root / "outputs" / "3dgs" / cls / mdl
        sugar = root / "outputs" / "sugar" / cls / mdl
        input_count = len(media_files(scene / "input"))
        image_count = len(media_files(scene / "images"))
        has_video = bool(media_files(scene / "raw_video", VIDEO_EXTENSIONS))
        dgs_done = has_checkpoint(dgs, 30000)
        sugar_done = has_sugar_result(sugar)
        evaluated = all(has_evaluation(out) for out in (dgs, sugar))
        result.setdefault(cls, {})[mdl] = {
            "input_count": input_count, "images_count": input_count or image_count,
            "undistorted_count": image_count, "has_video": has_video,
            "has_gt_lidar": (scene / "ground_truth" / "lidar.ply").is_file(),
            "colmap_done": has_colmap_model(scene), "3dgs_7k": has_checkpoint(dgs, 7000),
            "3dgs_30k": dgs_done, "sugar_mesh": any((sugar / "mesh").glob("*.ply")),
            "sugar_obj": sugar_done, "evaluated": evaluated,
            "input_path": f"datasets/{cls}/{mdl}/input/",
            "output_path": f"outputs/{{3dgs,sugar}}/{cls}/{mdl}/",
            "overall_status": "Completed" if dgs_done and sugar_done and evaluated else
                              "Trained" if dgs_done and sugar_done else
                              "Ready" if input_count or image_count or has_video else "No Data"}
    return result


def get_metrics_table(root=PROJECT_ROOT):
    path = Path(root) / "outputs/logs/evaluation_reports/summary_comparison.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def get_preview_images(cls, mdl, root=PROJECT_ROOT):
    scene_pairs(cls, mdl)
    items = []
    for method in ("3dgs", "sugar"):
        folder = Path(root) / "outputs" / method / cls / mdl / "evaluation" / "renders"
        for index, path in enumerate(media_files(folder)[:3]):
            items.append({"src": "/static/" + path.relative_to(root).as_posix(),
                          "label": f"{method} test view #{index + 1}", "type": method})
    return {"images": items}


def build_command(job, root=PROJECT_ROOT):
    root = Path(root)
    cls, mdl, action = job["class_name"], job["model_id"], job["action"]
    scene_pairs(cls, mdl)
    scene = root / "datasets" / cls / mdl
    common = ["--statue_class", cls, "--model_id", mdl]
    gpu = ["--gpu", str(job.get("gpu", 0))]
    settings = resolution_settings(job)
    image_args = ['--image_downscale', str(settings['image_downscale']),
                  '--image_max_size', str(settings['image_max_size'])]
    if action == "full_pipeline":
        script = "batch_run_all.py"
        args = ["--class_name", cls, "--model_id", mdl, *gpu,
                *sugar_arguments(job)]
        if not job.get("resume", True):
            args.append("--no_resume")
    elif action == "extract_frames":
        videos = media_files(scene / "raw_video", VIDEO_EXTENSIONS)
        if not videos:
            raise ValueError("Tidak ada video pada raw_video/.")
        if media_files(scene / "input"):
            raise ValueError("input/ sudah berisi foto. Gunakan folder kosong untuk ekstraksi agar foto tidak tertimpa.")
        script, args = "01_extract_frames.py", ["-v", str(videos[0]), "-o", str(scene / "input"), *common]
    elif action == "colmap":
        script, args = "02_run_colmap.py", ["-s", str(scene), *common, *gpu]
    elif action == "3dgs":
        script, args = "03_train_3dgs.py", ["-s", str(scene), *common, *gpu]
    elif action == "sugar":
        script, args = "04_train_sugar.py", ["-s", str(scene), "-c",
            str(root / "outputs/3dgs" / cls / mdl), *common, *gpu,
            *sugar_arguments(job)]
    elif action == "evaluate_visual":
        script, args = "06_evaluate_rendering.py", [*common, *gpu, "--method", job["evaluation_method"]]
    elif action == "evaluate":
        method = job["evaluation_method"]
        if method == "3DGS":
            model = root / "outputs/3dgs" / cls / mdl / "point_cloud/iteration_30000/point_cloud.ply"
        else:
            output = root / "outputs/sugar" / cls / mdl
            manifest = json.loads((output / "pipeline_result.json").read_text(encoding="utf-8"))
            model = output / manifest["textured_mesh"]
        script, args = "05_evaluate_comparison.py", [*common, "--method", method, "-p", str(model),
            "-g", str(scene / "ground_truth/lidar.ply"), "--scale_factor", str(job["scale_factor"])]
    else:
        raise ValueError("Tahap pipeline tidak dikenal.")
    if action in {'full_pipeline', 'colmap'}:
        args.extend(['--colmap_max_image_size', str(settings['colmap_max_image_size'])])
        args.extend(colmap_arguments(job))
    if action in {'full_pipeline', '3dgs', 'sugar', 'evaluate_visual'}:
        args.extend(image_args)
    return [sys.executable, "-u", str(root / "pipeline_tools" / script), *args]


def prepare_jobs(data, root=PROJECT_ROOT):
    if not isinstance(data, dict):
        raise ValueError("Body harus berupa objek JSON.")
    action = data.get("action", "full_pipeline")
    if action not in {"full_pipeline", "extract_frames", "colmap", "3dgs", "sugar", "evaluate", "evaluate_visual"}:
        raise ValueError("Tahap pipeline tidak dikenal.")
    cls, mdl = data.get("class_name", ""), data.get("model_id", "")
    if not isinstance(cls, str) or not isinstance(mdl, str):
        raise ValueError("Kelas dan model harus berupa teks.")
    if action != "full_pipeline" and (not cls or not mdl):
        raise ValueError("Pilih satu kelas dan satu model untuk tahap ini.")
    pairs = scene_pairs(cls, mdl)
    settings = {**resolution_settings(data), **colmap_settings(data), **sugar_settings(data)}
    references = research_presets()
    selected_sugar = sugar_settings(data)
    source_preset = next((key for key, preset in references['presets'].items()
                          if preset['values'] == selected_sugar), 'custom')
    try:
        raw_gpu = data.get('gpu', 0)
        if isinstance(raw_gpu, bool) or not isinstance(raw_gpu, (str, int)):
            raise ValueError('GPU harus berupa bilangan bulat.')
        gpu = int(raw_gpu)
        vertices = settings['vertices']
    except (TypeError, ValueError):
        raise ValueError("GPU dan target mesh harus berupa bilangan bulat.")
    if gpu < 0 or vertices < 1000 or vertices > 2000000:
        raise ValueError("GPU harus >= 0; target mesh antara 1000 dan 2000000.")
    refinement = settings['refinement_time']
    if refinement not in {"short", "medium", "long"}:
        raise ValueError("Refinement harus short, medium, atau long.")
    method = data.get("evaluation_method", "SuGaR")
    if method not in {"3DGS", "SuGaR"}:
        raise ValueError("Metode evaluasi harus 3DGS atau SuGaR.")
    scale = None
    if action == "evaluate":
        try:
            scale = float(data.get("scale_factor"))
        except (TypeError, ValueError):
            raise ValueError("Masukkan faktor kalibrasi skala numerik.")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Faktor skala harus finite dan > 0.")
    jobs, skipped = [], []
    for cls, mdl in pairs:
        if action == "full_pipeline" and not has_data(Path(root) / "datasets" / cls / mdl):
            skipped.append(f"{cls}/{mdl}")
            continue
        jobs.append(dict(action=action, class_name=cls, model_id=mdl, gpu=gpu,
                         resume=data.get("resume", True) is True,
                         evaluation_method=method, scale_factor=scale,
                         sugar_configuration_source=source_preset,
                         upstream_reference=references['source'], **settings))
    return jobs, skipped


class DashboardHandler(BaseHTTPRequestHandler):
    @property
    def root(self):
        return getattr(self.server, 'project_root', PROJECT_ROOT)

    @property
    def queue(self):
        return self.server.job_queue

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                self.send_file(self.root / "web_dashboard/index.html", "text/html; charset=utf-8")
            elif path == "/dashboard.js":
                self.send_file(self.root / "web_dashboard/dashboard.js", "text/javascript; charset=utf-8")
            elif path == "/api/catalog":
                self.send_json({"classes": CATALOG, "total_models": len(scene_pairs())})
            elif path == "/api/status":
                self.send_json(get_project_status(self.root))
            elif path == '/api/colmap':
                self.send_json(inspect_colmap())
            elif path == '/api/presets':
                self.send_json(research_presets())
            elif path == "/api/metrics":
                self.send_json(get_metrics_table(self.root))
            elif path == "/api/metrics.csv":
                self.send_file(self.root / "outputs/logs/evaluation_reports/summary_comparison.csv", "text/csv; charset=utf-8", download=True)
            elif path in ("/api/logs", "/api/queue"):
                self.send_json(self.queue.snapshot())
            elif path.startswith('/api/jobs/') and path.endswith('/logs'):
                parts = path.strip('/').split('/')
                if len(parts) != 4:
                    raise ValueError('Format ID pekerjaan tidak valid.')
                self.send_json(self.queue.job_logs(parts[2]))
            elif path.startswith("/api/previews/"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    raise ValueError("Format preview tidak valid.")
                self.send_json(get_preview_images(parts[2], parts[3], self.root))
            elif path.startswith("/static/"):
                file = (self.root / path[len("/static/"):]).resolve()
                output = (self.root / "outputs").resolve()
                if output not in file.parents:
                    self.send_error(403)
                else:
                    self.send_file(file)
            else:
                self.send_error(404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 <= length <= 65536:
                raise ValueError("Body terlalu besar.")
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(data, dict):
                raise ValueError("Body harus berupa objek JSON.")
            if path == "/api/run":
                jobs, skipped = prepare_jobs(data, self.root)
                if not jobs:
                    raise ValueError("Belum ada foto/video pada dataset yang dipilih. Isi input/ atau raw_video/ terlebih dahulu.")
                added, duplicates = self.queue.enqueue(jobs)
                self.send_json({"message": f"{len(added)} pekerjaan ditambahkan ke antrean.",
                    "added": added, "duplicates": duplicates, "skipped_empty": skipped}, 202)
            elif path == "/api/stop":
                self.queue.stop()
                self.send_json({"message": "Proses dihentikan; antrean dijeda. Klik Lanjutkan untuk pekerjaan berikutnya."})
            elif path in ("/api/queue/pause", "/api/queue/resume"):
                self.queue.set_paused(path.endswith("pause"))
                self.send_json({"message": "Antrean dijeda setelah pekerjaan aktif selesai." if path.endswith("pause") else "Antrean dilanjutkan."})
            elif path == "/api/queue/cancel":
                job_id = data.get("job_id")
                if job_id is not None and not isinstance(job_id, str):
                    raise ValueError("ID pekerjaan tidak valid.")
                self.send_json({"cancelled": self.queue.cancel_pending(job_id)})
            else:
                self.send_error(404)
        except (ValueError, TypeError, UnicodeError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def send_file(self, path, mime=None, download=False):
        if not path.is_file():
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", mime or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        if download:
            self.send_header("Content-Disposition", 'attachment; filename="summary_comparison.csv"')
        self.end_headers()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def send_json(self, data, status=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def run_server(port=8080, host="127.0.0.1"):
    with gpu_lock("dashboard", wait=False):
        initialize_workspace()
        httpd = HTTPServer((host, port), DashboardHandler)
        queue = JobQueue(PROJECT_ROOT, build_command)
        httpd.job_queue = queue
        print(f"Dashboard 18 model: http://{host}:{port} | satu worker | Ctrl+C untuk berhenti", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            queue.close()
            httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int, nargs="?", default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run_server(args.port, args.host)
