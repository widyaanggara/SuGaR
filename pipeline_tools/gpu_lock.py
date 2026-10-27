"""OS-released advisory GPU lock shared by the project's training entry points."""
import os
import time
from contextlib import contextmanager
from pathlib import Path

from pipeline_tools.dataset_registry import PROJECT_ROOT


@contextmanager
def gpu_lock(gpu=0, root=PROJECT_ROOT, wait=True):
    folder = Path(root) / "outputs" / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"gpu_{gpu}.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        warned = False
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if not wait:
                    raise RuntimeError(f"Lock {gpu} sudah digunakan proses lain.")
                if not warned:
                    print(f"[QUEUE] Menunggu GPU {gpu} dipakai pipeline lain...", flush=True)
                    warned = True
                time.sleep(1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
