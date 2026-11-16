"""Delete one catalogue slot while holding the queue's worker lock."""
import csv
import io
import json
import shutil
from pathlib import Path

from pipeline_tools.dataset_registry import CATALOG


def delete_model_data(queue, data):
    cls, mdl = data.get('class_name'), data.get('model_id')
    if not isinstance(cls, str) or not isinstance(mdl, str) or mdl not in CATALOG.get(cls, ()):
        raise ValueError('Pilih satu patung yang terdaftar.')
    if data.get('confirmation') != f'{cls}/{mdl}':
        raise ValueError('Ketik ID patung lengkap untuk mengonfirmasi penghapusan.')
    root = queue.root.resolve()

    def checked(path):
        path = Path(path)
        if not path.resolve().is_relative_to(root) or path.resolve() == root:
            raise ValueError('Lokasi data berada di luar workspace; penghapusan dibatalkan.')
        # Reject links/junctions, including inside trees, rather than following them.
        for item in (path, *path.parents):
            if item == root:
                break
            if item.is_symlink() or (hasattr(item, 'is_junction') and item.is_junction()):
                raise ValueError('Folder data berupa link/junction; penghapusan dibatalkan.')
        return path

    with queue.condition:
        if queue.active_id or any(j['status'] in ('RUNNING', 'STOPPING') for j in queue.jobs):
            raise ValueError('Hentikan pipeline dan tunggu proses selesai sebelum menghapus data.')
        paths = [root / 'datasets' / cls / mdl]
        for base in ('outputs', 'outputs/ablations/scaffold_anchor'):
            for method in ('3dgs', 'sugar', '2dgs', 'radegs', 'scaffold', 'scaffold_gs', '3dgs_scaffold_anchor', 'sugar_scaffold_anchor'):
                paths.append(root / base / method / cls / mdl)
        paths.extend((root / 'outputs/logs/training_logs').glob(f'{cls}_{mdl}_*'))
        jobs = [j for j in queue.jobs if (j['class_name'], j['model_id']) == (cls, mdl)]
        paths.extend(queue.folder / 'logs' / (j['id'] + '.log') for j in jobs)
        updates = {}
        for name in ('model_names', 'geometry_scales', 'geometry_settings'):
            path = checked(root / 'datasets' / (name + '.json'))
            if path.exists():
                values = json.loads(path.read_text(encoding='utf-8'))
                values.get(cls, {}).pop(mdl, None)
                if cls in values and not values[cls]:
                    del values[cls]
                updates[path] = json.dumps(values, ensure_ascii=False, indent=2) + '\n'
        report = checked(root / 'outputs/logs/evaluation_reports/summary_comparison.csv')
        markdown = checked(report.with_suffix('.md'))
        if report.exists():
            with report.open(encoding='utf-8', newline='') as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                if not fields or not {'Kelas', 'Model ID'}.issubset(fields):
                    raise ValueError('Kolom rekap evaluasi tidak dikenal; penghapusan dibatalkan.')
                rows = [r for r in reader if (r['Kelas'], r['Model ID']) != (cls, mdl)]
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            updates[report] = stream.getvalue()
            updates[markdown] = '# Tabel Rekapitulasi Hasil Komparasi\n\n' + '\n'.join(
                ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |'] +
                ['| ' + ' | '.join(str(r.get(f, '')).replace('|', '\\|') for f in fields) + ' |' for r in rows]) + '\n'
        elif markdown.exists():
            raise ValueError('Rekap CSV tidak ditemukan; periksa rekap Markdown sebelum menghapus.')
        for path in paths:
            checked(path)
            if path.is_dir():
                for child in path.rglob('*'):
                    checked(child)
        for path in updates:
            checked(path.with_suffix(path.suffix + '.tmp'))
        # Remove queued retries before deleting files so an I/O failure cannot restart them.
        queue.jobs = [j for j in queue.jobs if j not in jobs]
        queue._save()
        for j in jobs:
            queue.buffers.pop(j['id'], None)
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for path, content in updates.items():
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)
        for folder in ('input', 'images', 'raw_video', 'ground_truth'):
            (root / 'datasets' / cls / mdl / folder).mkdir(parents=True, exist_ok=True)
        return {'message': f'Semua data {cls}/{mdl} telah dihapus. Slot siap diisi kembali.'}
