"""Persistent FIFO with exactly one subprocess tree active at a time."""
import collections
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path


def process_options():
    return {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def terminate_process_tree(process):
    """Wait for the old process tree before permitting another GPU job."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            process.poll()  # Reap the leader, but also wait for its children.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


class JobQueue:
    def __init__(self, root, command_builder, start_worker=True):
        self.root = Path(root)
        self.folder = self.root / "outputs" / "queue"
        (self.folder / "logs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.folder / "state.json"
        self.command_builder = command_builder
        self.condition = threading.Condition(threading.RLock())
        self.jobs = []
        self.paused = False
        self.closing = False
        self.process = None
        self.active_id = None
        self.cancelling = False
        self.buffers = {}
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.jobs = data["jobs"]
            for job in self.jobs:
                if job["status"] in ("RUNNING", "STOPPING"):
                    job.update(status="INTERRUPTED", finished_at=time.time(),
                               error="Server berhenti. Periksa GPU lalu tambahkan ulang pekerjaan ini.")
            # A server restart never automatically reruns a potentially orphaned process.
            self.paused = True
            self._save()
        self.worker = threading.Thread(target=self._work, name="pipeline-queue", daemon=True)
        if start_worker:
            self.worker.start()

    def _save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"paused": self.paused, "jobs": self.jobs}, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def enqueue(self, specs):
        added, duplicates = [], []
        with self.condition:
            def key_for(item):
                return (item['action'], item['class_name'], item['model_id'],
                        item.get('evaluation_method', '') if item['action'].startswith('evaluate') else '')
            occupied = {key_for(j) for j in self.jobs
                        if j["status"] in ("QUEUED", "RUNNING", "STOPPING")}
            for spec in specs:
                key = key_for(spec)
                if key in occupied:
                    duplicates.append("/".join(key))
                    continue
                job = dict(spec, id=uuid.uuid4().hex, status="QUEUED", created_at=time.time(),
                           started_at=None, finished_at=None, error=None, returncode=None)
                self.jobs.append(job)
                added.append(job["id"])
                occupied.add(key)
            self._save()
            self.condition.notify_all()
        return added, duplicates

    def _log(self, job, line):
        with self.condition:
            self.buffers.setdefault(job["id"], collections.deque(maxlen=300)).append(line)

    def snapshot(self):
        with self.condition:
            active = next((j for j in self.jobs if j["id"] == self.active_id), None)
            latest = active or (self.jobs[-1] if self.jobs else None)
            if not active:
                finished = [j for j in self.jobs if j.get("finished_at")]
                latest = max(finished, key=lambda j: j["finished_at"]) if finished else latest
            logs = list(self.buffers.get(latest["id"], [])) if latest else []
            if latest and not logs:
                path = self.folder / "logs" / (latest["id"] + ".log")
                if path.exists():
                    with path.open(encoding="utf-8", errors="replace") as stream:
                        logs = list(collections.deque(stream, maxlen=300))
            counts = dict(collections.Counter(j["status"] for j in self.jobs))
            return {"paused": self.paused, "counts": counts,
                    "jobs": [dict(j) for j in self.jobs[-200:]],
                    "is_running": active is not None,
                    "action": latest["action"] if latest else None,
                    "target": f"{latest['class_name']}/{latest['model_id']}" if latest else None,
                    "status": active["status"] if active else ("PAUSED" if self.paused else latest["status"] if latest else "IDLE"),
                    "elapsed_seconds": round(time.time() - active["started_at"], 1) if active else 0,
                    "logs": logs}

    def set_paused(self, paused):
        with self.condition:
            self.paused = paused
            self._save()
            self.condition.notify_all()

    def job_logs(self, job_id):
        with self.condition:
            job = next((item for item in self.jobs if item['id'] == job_id), None)
            if job is None:
                raise ValueError('Pekerjaan tidak ditemukan.')
            lines = list(self.buffers.get(job_id, []))
            path = self.folder / 'logs' / (job_id + '.log')
            if not lines and path.is_file():
                with path.open(encoding='utf-8', errors='replace') as stream:
                    lines = list(collections.deque(stream, maxlen=300))
            return {'job': dict(job), 'logs': lines}

    def cancel_pending(self, job_id=None):
        with self.condition:
            count = 0
            for job in self.jobs:
                if job["status"] == "QUEUED" and (job_id is None or job["id"] == job_id):
                    job.update(status="CANCELLED", finished_at=time.time())
                    count += 1
            self._save()
            return count

    def stop(self):
        # Pause first so the worker cannot start the next model during cleanup.
        with self.condition:
            self.paused = True
            job = next((j for j in self.jobs if j["id"] == self.active_id), None)
            if job:
                job["status"] = "STOPPING"
                self.cancelling = True
            process = self.process
            self._save()
        if process is not None:
            terminate_process_tree(process)
        with self.condition:
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.active_id is None, timeout=15)

    def close(self):
        with self.condition:
            self.closing = True
        self.stop()
        with self.condition:
            self.condition.notify_all()
        if self.worker.is_alive():
            self.worker.join(timeout=15)

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closing or (not self.paused and any(
                    j["status"] == "QUEUED" for j in self.jobs)))
                if self.closing:
                    return
                job = next(j for j in self.jobs if j["status"] == "QUEUED")
                job.update(status="RUNNING", started_at=time.time())
                self.active_id = job["id"]
                self.cancelling = False
                self._save()
            log_path = self.folder / "logs" / (job["id"] + ".log")
            try:
                command = self.command_builder(job)
                with log_path.open("a", encoding="utf-8", buffering=1) as stream:
                    header = "[QUEUE] " + subprocess.list2cmdline(command)
                    stream.write(header + "\n")
                    self._log(job, header)
                    with self.condition:
                        if self.cancelling or self.closing:
                            raise InterruptedError("Pekerjaan dibatalkan sebelum proses dimulai.")
                        self.process = subprocess.Popen(command, cwd=str(self.root),
                            env=dict(os.environ, PYTHONUNBUFFERED="1"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", **process_options())
                        process = self.process
                    for line in process.stdout:
                        stream.write(line)
                        self._log(job, line.rstrip())
                    process.wait()
                    process.stdout.close()
                    with self.condition:
                        job["returncode"] = process.returncode
                        job["status"] = "CANCELLED" if self.cancelling else "COMPLETED" if process.returncode == 0 else "FAILED"
                        if job["status"] == "FAILED":
                            job["error"] = f"Proses keluar dengan kode {process.returncode}. Lihat log."
            except Exception as exc:
                with self.condition:
                    job.update(status="CANCELLED" if self.cancelling else "FAILED", error=str(exc))
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"[ERROR] {exc}\n")
                self._log(job, f"[ERROR] {exc}")
            finally:
                # An I/O/logging exception must not leave a training child alive.
                if self.process is not None and self.process.poll() is None:
                    terminate_process_tree(self.process)
                with self.condition:
                    job["finished_at"] = time.time()
                    self.process = None
                    self.active_id = None
                    self.cancelling = False
                    self._save()
                    # Keep only the current/recent log buffer in RAM; full logs are on disk.
                    self.buffers = {job["id"]: self.buffers.get(job["id"], collections.deque(maxlen=300))}
                    self.condition.notify_all()
