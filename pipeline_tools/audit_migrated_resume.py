"""Read-only migration audit. Does not import GPU workers or start a queue.

This inventories recovery metadata; it does not prove optimizer resumability.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools.dataset_registry import has_checkpoint, has_sugar_result, has_evaluation
from pipeline_tools.repair_migrated_sugar import signature


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root, source, target):
    root, source, target = Path(root).resolve(), Path(source).resolve(), Path(target).resolve()
    if not source.is_dir() or not target.is_dir() or source == target:
        raise ValueError('Sumber dan tujuan harus ada dan berbeda.')
    jobs = read(target / 'queue/state.json').get('jobs', [])
    latest = {}
    for job in jobs:
        key = tuple(str(job.get(k, '')) for k in ('experiment', 'class_name', 'model_id', 'action'))
        latest[key] = job
    print('=== Status terakhir per pekerjaan (riwayat antrean, bukan bukti hasil) ===')
    for key, job in sorted(latest.items()):
        if job.get('status') != 'COMPLETED':
            print('/'.join(key), '|', job.get('status'), '| id:', job.get('id'))
    totals = Counter()
    for old_identity in sorted(source.rglob('recovery/identity.json')):
        relative = old_identity.parent.parent.relative_to(source)
        original, output = source / relative, target / relative
        method = relative.parts[-3] if len(relative.parts) >= 3 else ''
        print('\n===', relative, '===', flush=True)
        try:
            old_data = read(old_identity)
            identity = read(output / 'recovery/identity.json')
            context = identity['context']
            print('Identitas sama dengan cadangan:', old_data == identity)
            if method in ('3dgs', 'sugar'):
                completed = has_checkpoint(output, 30000) if method == '3dgs' else has_sugar_result(output)
                print('Artefak akhir:', completed, '| Evaluasi:', has_evaluation(output))
                totals['artefak_lengkap' if completed else 'artefak_belum_lengkap'] += 1
            differences = []
            for name, expected in context.get('code', {}).items():
                path = root / name
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    differences.append(name)
            print('Kode training berbeda:', ', '.join(differences) or 'tidak ada')
            scene = context.get('scene', {})
            scene_path = Path(scene.get('scene', '__missing_scene__'))
            dataset_bad = 0
            for group, subdir in (('images', 'images'), ('colmap', 'sparse/0')):
                for name, expected in scene.get(group, {}).items():
                    path = scene_path / subdir / name
                    if not path.is_file() or signature(path) != expected:
                        dataset_bad += 1
            print('Metadata dataset tidak cocok:', dataset_bad)
            if context.get('baseline'):
                baseline = target / relative.parent.parent.parent / '3dgs' / relative.parts[-2] / relative.parts[-1] / 'point_cloud/iteration_7000/point_cloud.ply'
                print('Baseline cocok dengan identitas:', baseline.is_file() and signature(baseline) == context['baseline'])
            print('Identitas menyimpan reused_coarse:', bool(context.get('reused_coarse')))
            slots = Counter()
            for old_meta in sorted((original / 'recovery').rglob('*.json')):
                if old_meta.name not in ('a.json', 'b.json'):
                    continue
                slot_relative = old_meta.relative_to(source)
                new_meta = target / slot_relative
                old_record, new_record = read(old_meta), read(new_meta)
                old_pt, new_pt = old_meta.with_suffix('.pt'), new_meta.with_suffix('.pt')
                if not old_pt.is_file() or not new_pt.is_file():
                    slots['checkpoint_hilang'] += 1
                    continue
                if old_record.get('signature') != signature(old_pt):
                    slots['cadangan_tidak_cocok'] += 1
                elif new_record.get('signature') != signature(new_pt):
                    slots['signature_F_berbeda'] += 1
                else:
                    slots['signature_cocok'] += 1
                if old_record.get('identity') != new_record.get('identity') or old_record.get('fingerprint') != new_record.get('fingerprint'):
                    slots['identitas_slot_berbeda'] += 1
            print('Slot resume:', dict(slots))
            totals.update(slots)
            stages = list((output / 'recovery/stages').glob('*/*.json'))
            print('Catatan tahap:', len(stages))
            totals['run'] += 1
            totals['run_kode_berbeda'] += bool(differences)
            totals['run_dataset_berbeda'] += bool(dataset_bad)
        except (OSError, ValueError, KeyError, TypeError) as error:
            totals['error'] += 1
            print('ERROR:', str(error))
    print('\nRINGKASAN:', dict(totals))
    print('READ ONLY. Kecocokan konfigurasi pekerjaan baru, isi optimizer, fingerprint tahap dan metode eksternal belum divalidasi penuh.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--source', default='outputs_backup_lama')
    parser.add_argument('--target', default='outputs')
    args = parser.parse_args()
    try:
        audit(args.root, args.source, args.target)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'STOP: ' + str(error) + '\n')
