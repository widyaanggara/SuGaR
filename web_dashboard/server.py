"""Local dashboard and persistent, single-worker pipeline queue."""
import argparse
import csv
import io
import json
import mimetypes
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline_tools.dataset_registry import (CATALOG, VIDEO_EXTENSIONS, has_checkpoint,
    has_colmap_model, has_data, has_sugar_result, has_evaluation, media_files, scene_pairs)
from pipeline_tools.init_dataset_folders import initialize_workspace
from pipeline_tools.gpu_lock import gpu_lock
from web_dashboard.job_queue import JobQueue
from web_dashboard.metrics_export import build_excel
from pipeline_tools.run_settings import (resolution_settings, colmap_settings, colmap_arguments,
    sugar_settings, sugar_arguments, research_presets)
from pipeline_tools.colmap_capabilities import inspect_colmap
from pipeline_tools.model_names import read_model_names, model_display_name, save_model_name
from pipeline_tools.geometry_config import (read_geometry_scales, save_geometry_scales, positive_scale,
    geometry_model_path, geometry_source_ready, geometry_result_current, nonempty_file,
    read_geometry_settings, save_geometry_settings, geometry_options)


def get_project_status(root=PROJECT_ROOT):
    root = Path(root)
    names = read_model_names(root)
    scales = read_geometry_scales(root)
    geometry_settings = read_geometry_settings(root)
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
        scale = scales.get(cls, {}).get(mdl)
        options = geometry_settings.get(cls, {}).get(mdl)
        geometry_methods = {method: geometry_result_current(root, cls, mdl, method, scale, options)
                            for method in ('3DGS', 'SuGaR')}
        result.setdefault(cls, {})[mdl] = {
            "display_name": model_display_name(cls, mdl, names),
            "custom_name": names.get(cls, {}).get(mdl, ""),
            "input_count": input_count, "images_count": input_count or image_count,
            "undistorted_count": image_count, "has_video": has_video,
            "has_gt_lidar": nonempty_file(scene / "ground_truth/lidar.ply"),
            "geometry_scale_factor": scale,
            "geometry_settings": options,
            "geometry_sources": {method: geometry_source_ready(root, cls, mdl, method)
                                 for method in ('3DGS', 'SuGaR')},
            "geometry_methods": geometry_methods, "geometry_evaluated": all(geometry_methods.values()),
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
    names = read_model_names(root)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['Nama Model'] = model_display_name(row.get('Kelas', ''), row.get('Model ID', ''), names)
    return rows


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
        if job.get('pipeline_start') == 'existing_colmap':
            args.append('--skip_colmap')
        if job.get('experiment', 'baseline') == 'scaffold_anchor':
            args.extend(['--experiment', 'scaffold_anchor', '--scaffold_repo', job['scaffold_repo'],
                         '--scaffold_python', job.get('scaffold_python') or sys.executable,
                         '--scaffold_iterations', str(job.get('scaffold_iterations', 30000)),
                         '--scaffold_resolution', str(job.get('scaffold_resolution', 1))])
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
        model = geometry_model_path(root, cls, mdl, method)
        script, args = "05_evaluate_comparison.py", [*common, "--method", method, "-p", str(model),
            "-g", str(scene / "ground_truth/lidar.ply")]
        options = job.get('geometry_options')
        if options:
            args.extend(['--lidar_unit', options['lidar_unit']])
        if options and options['alignment_mode'] == 'automatic':
            args.append('--auto_align')
        else:
            args.extend(['--scale_factor', str(job['scale_factor'])])
        if job.get('apply_icp', True) is False:
            args.append('--no_icp')
    else:
        raise ValueError("Tahap pipeline tidak dikenal.")
    if action in {'full_pipeline', 'colmap'}:
        args.extend(['--colmap_max_image_size', str(settings['colmap_max_image_size'])])
        args.extend(colmap_arguments(job))
    if action in {'full_pipeline', '3dgs', 'sugar', 'evaluate_visual'}:
        args.extend(image_args)
    if action in {'3dgs', 'sugar'} and job.get('resume', True) is False:
        args.append('--no_resume')
    return [sys.executable, "-u", str(root / "pipeline_tools" / script), *args]


def prepare_jobs(data, root=PROJECT_ROOT):
    if not isinstance(data, dict):
        raise ValueError("Body harus berupa objek JSON.")
    action = data.get("action", "full_pipeline")
    experiment = data.get('experiment', 'baseline')
    if experiment not in ('baseline', 'scaffold_anchor'):
        raise ValueError('Eksperimen harus baseline atau scaffold_anchor.')
    scaffold = {}
    if experiment == 'scaffold_anchor':
        if action != 'full_pipeline':
            raise ValueError('Eksperimen Scaffold dijalankan melalui Pipeline lengkap.')
        repo = data.get('scaffold_repo')
        if not isinstance(repo, str) or not repo.strip() or not (Path(repo) / 'train.py').is_file():
            raise ValueError('Isi lokasi repository Scaffold-GS pada komputer training (harus memuat train.py).')
        python = data.get('scaffold_python') or sys.executable
        if not isinstance(python, str):
            raise ValueError('Python Scaffold harus berupa path executable.')
        iterations = data.get('scaffold_iterations', 30000)
        resolution = data.get('scaffold_resolution', 1)
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError('Iterasi Scaffold harus bilangan bulat positif.')
        if isinstance(resolution, bool) or resolution not in (1, 2, 4, 8):
            raise ValueError('Downscale Scaffold harus 1, 2, 4, atau 8.')
        scaffold = dict(scaffold_repo=str(Path(repo).resolve()), scaffold_python=python,
                        scaffold_iterations=iterations, scaffold_resolution=resolution)
    if action not in {"full_pipeline", "extract_frames", "colmap", "3dgs", "sugar", "evaluate", "evaluate_visual"}:
        raise ValueError("Tahap pipeline tidak dikenal.")
    pipeline_start = data.get('pipeline_start', 'auto')
    if not isinstance(pipeline_start, str) or pipeline_start not in ('auto', 'existing_colmap'):
        raise ValueError('Awal pipeline harus auto atau existing_colmap.')
    if pipeline_start != 'auto' and action != 'full_pipeline':
        raise ValueError('Awal pipeline hanya berlaku untuk Pipeline lengkap.')
    cls, mdl = data.get("class_name", ""), data.get("model_id", "")
    if not isinstance(cls, str) or not isinstance(mdl, str):
        raise ValueError("Kelas dan model harus berupa teks.")
    if action not in {"full_pipeline", "evaluate_visual", "evaluate"} and (not cls or not mdl):
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
    allowed_methods = {'3DGS', 'SuGaR', 'both'} if action in {'evaluate_visual', 'evaluate'} else {'3DGS', 'SuGaR'}
    if not isinstance(method, str) or method not in allowed_methods:
        raise ValueError("Metode evaluasi harus 3DGS, SuGaR, atau keduanya untuk evaluasi.")
    methods = ('3DGS', 'SuGaR') if method == 'both' else (method,)
    apply_icp = data.get('apply_icp', True)
    if not isinstance(apply_icp, bool):
        raise ValueError('Pilihan ICP harus berupa boolean.')
    require_geometry_settings = data.get('geometry_settings_required', False)
    if not isinstance(require_geometry_settings, bool):
        raise ValueError('Pilihan pengaturan geometri wajib harus berupa boolean.')
    explicit_scale, scales, geometry_settings = None, {}, {}
    if action == "evaluate":
        if 'scale_factor' in data:
            if len(pairs) != 1:
                raise ValueError('Evaluasi semua model harus memakai skala kalibrasi per model, bukan satu skala bersama.')
            explicit_scale = positive_scale(data['scale_factor'])
        scales = read_geometry_scales(root)
        if require_geometry_settings:
            scales = {}
        geometry_settings = read_geometry_settings(root)
        if 'geometry_options' in data:
            selected_options = geometry_options(data['geometry_options'])
            geometry_settings = {cls: {mdl: selected_options for c, mdl in pairs if c == cls}
                                 for cls, _ in pairs}
    jobs, skipped = [], []
    for cls, mdl in pairs:
        if action == "full_pipeline" and not has_data(Path(root) / "datasets" / cls / mdl):
            skipped.append(f"{cls}/{mdl}")
            continue
        if action == 'full_pipeline' and pipeline_start == 'existing_colmap' and not has_colmap_model(Path(root) / 'datasets' / cls / mdl):
            skipped.append(f'{cls}/{mdl}: images/ hasil undistorsi dan sparse/0 COLMAP belum lengkap')
            continue
        scale = explicit_scale if explicit_scale is not None else scales.get(cls, {}).get(mdl)
        options = None if explicit_scale is not None else geometry_settings.get(cls, {}).get(mdl)
        if options and options['alignment_mode'] == 'manual':
            scale = options['scale_factor']
        if options and options['alignment_mode'] == 'automatic' and not apply_icp:
            raise ValueError('Penyelarasan otomatis tidak dapat digabungkan dengan --no_icp.')
        for current_method in methods:
            reason = ''
            if action == 'evaluate':
                if not nonempty_file(Path(root) / 'datasets' / cls / mdl / 'ground_truth/lidar.ply'):
                    reason = 'ground_truth/lidar.ply belum tersedia'
                elif scale is None and options is None:
                    reason = 'pilih satuan LiDAR dan penyelarasan otomatis, atau isi kalibrasi manual'
                elif not geometry_source_ready(root, cls, mdl, current_method):
                    reason = f'hasil rekonstruksi {current_method} belum tersedia'
            elif action == 'evaluate_visual':
                baseline = Path(root) / 'outputs/3dgs' / cls / mdl
                ready = (has_checkpoint(baseline, 30000) if current_method == '3DGS' else
                         has_checkpoint(baseline, 7000) and has_sugar_result(Path(root) / 'outputs/sugar' / cls / mdl))
                if not ready:
                    reason = f'checkpoint evaluasi {current_method} belum lengkap'
            if reason:
                skipped.append(f'{cls}/{mdl} ({current_method}): {reason}')
                continue
            jobs.append(dict(action=action, class_name=cls, model_id=mdl, gpu=gpu,
                             experiment=experiment, **scaffold,
                             pipeline_start=pipeline_start,
                             resume=data.get("resume", True) is True,
                             evaluation_method=current_method, scale_factor=scale, apply_icp=apply_icp,
                             geometry_options=options,
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
                self.send_metrics_csv()
            elif path == "/api/metrics.xlsx":
                payload = build_excel(get_metrics_table(self.root), self.root)
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                self.send_header('Content-Disposition', 'attachment; filename="evaluasi_baseline_dan_ablasi.xlsx"')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(payload)
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
        except OSError:
            self.send_json({"error": "File dashboard tidak dapat dibaca. Periksa izin dan ruang penyimpanan server."}, 500)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 <= length <= 65536:
                raise ValueError("Body terlalu besar.")
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if not isinstance(data, dict):
                raise ValueError("Body harus berupa objek JSON.")
            if path == '/api/geometry-scales':
                self.send_json({'scales': save_geometry_scales(data.get('scales'), self.root)})
            elif path == '/api/geometry-settings':
                self.send_json({'settings': save_geometry_settings(data.get('settings'), self.root)})
            elif path == "/api/model-name":
                result = save_model_name(data.get('class_name'), data.get('model_id'),
                                         data.get('display_name'), self.root)
                self.send_json(result)
            elif path == "/api/run":
                jobs, skipped = prepare_jobs(data, self.root)
                if not jobs and data.get('action', 'full_pipeline') == 'full_pipeline':
                    if data.get('pipeline_start') == 'existing_colmap':
                        raise ValueError('Belum ada target dengan hasil COLMAP lengkap. Siapkan images/ dan sparse/0 pada model yang dipilih.')
                    raise ValueError("Belum ada foto/video pada dataset yang dipilih. Isi input/ atau raw_video/ terlebih dahulu.")
                added, duplicates = self.queue.enqueue(jobs)
                self.send_json({"message": f"{len(added)} pekerjaan ditambahkan ke antrean.",
                    "added": added, "duplicates": duplicates, "skipped_empty": skipped, "skipped": skipped}, 202)
            elif path == "/api/stop":
                self.queue.stop()
                self.send_json({"message": "Proses dihentikan; antrean dijeda. Klik Lanjutkan untuk pekerjaan berikutnya."})
            elif path == '/api/queue/retry':
                if not isinstance(data.get('job_id'), str):
                    raise ValueError('ID pekerjaan wajib diisi.')
                self.send_json(self.queue.retry(data['job_id']), 202)
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
        except OSError:
            self.send_json({"error": "Perubahan tidak dapat disimpan. Periksa izin dan ruang penyimpanan server."}, 500)

    def send_metrics_csv(self):
        query = parse_qs(urlparse(self.path).query)
        experiment = query.get('experiment', ['all'])[0]
        method = query.get('method', ['all'])[0]
        if experiment not in ('all', 'baseline', 'scaffold_anchor') or method not in ('all', '3DGS', 'SuGaR'):
            self.send_json({'error': 'Filter evaluasi tidak valid.'}, 400)
            return
        path = self.root / 'outputs/logs/evaluation_reports/summary_comparison.csv'
        if not path.is_file():
            self.send_error(404)
            return
        with path.open(encoding='utf-8', newline='') as stream:
            fields = list(csv.DictReader(stream).fieldnames or [])
        if 'Nama Model' not in fields:
            fields.append('Nama Model')
        text = io.StringIO(newline='')
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        for row in get_metrics_table(self.root):
            row_method = row.get('Metode', '')
            allowed = {'baseline': ('3DGS', 'SuGaR'),
                       'scaffold_anchor': ('3DGS_scaffold_anchor', 'SuGaR_scaffold_anchor')}
            if experiment != 'all' and row_method not in allowed[experiment]:
                continue
            if method != 'all' and row_method not in (method, method + '_scaffold_anchor'):
                continue
            if row['Nama Model'].startswith(('=', '+', '-', '@')):
                row['Nama Model'] = "'" + row['Nama Model']
            writer.writerow(row)
        payload = text.getvalue().encode('utf-8-sig')
        self.send_response(200)
        self.send_header('Content-Type', 'text/csv; charset=utf-8')
        filename = 'summary_comparison.csv' if experiment == 'all' else f'summary_{experiment}.csv'
        self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(payload)

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
