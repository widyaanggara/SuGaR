"""Report allocator peaks from the actual training process, not its supervisor."""
import json
import os
from pathlib import Path
import sys


def write_training_peak():
    target = os.environ.get('TOGOGLAB_TRAINING_VRAM_REPORT')
    torch = sys.modules.get('torch')
    if not target or torch is None or not torch.cuda.is_initialized():
        return
    try:
        data = {'pid': os.getpid(), 'measurement': 'torch_peak_allocated',
                'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                'peak_reserved_bytes': int(torch.cuda.max_memory_reserved())}
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data), encoding='utf-8')
        temporary.replace(path)
        print(f'[VRAM] Training peak allocated={data["peak_allocated_bytes"] / 1024**3:.3f} GiB; '
              f'reserved={data["peak_reserved_bytes"] / 1024**3:.3f} GiB', flush=True)
    except Exception as exc:
        print(f'[VRAM] Gagal menyimpan pengukuran: {exc}', file=sys.stderr, flush=True)
