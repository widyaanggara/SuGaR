"""Read archived method logs independently of queue history."""
import re
from pathlib import Path

from pipeline_tools.dataset_registry import CATALOG


def archived_logs(root, class_name, model_id, method, session=''):
    methods = {'3DGS', 'SuGaR', '2DGS', 'RaDe-GS'}
    methods |= {name + '_scaffold_anchor' for name in tuple(methods)}
    if class_name not in CATALOG or model_id not in CATALOG[class_name] or method not in methods:
        raise ValueError('Patung atau metode tidak valid.')
    folder = Path(root) / 'outputs/logs/training_logs'
    prefix = f'{class_name}_{model_id}_{method}_'
    pattern = re.compile(re.escape(prefix) + r'\d{8}_\d{6}(?:_\d+)?\.log')
    files = sorted((p for p in folder.glob('*.log') if pattern.fullmatch(p.name)
                    and p.resolve().parent == folder.resolve()), key=lambda p: p.name, reverse=True)
    runs = [{'session': p.stem, 'label': p.stem[len(prefix):]} for p in files]
    if session and session not in {p.stem for p in files}:
        raise ValueError('Run log tidak ditemukan untuk patung dan metode ini.')
    selected = next((p for p in files if p.stem == session), files[0] if files else None)
    return {'runs': runs, 'session': selected.stem if selected else '',
            'text': selected.read_text(encoding='utf-8', errors='replace') if selected else ''}
