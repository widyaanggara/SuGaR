"""Verify migrated evaluation checkpoints; optionally rebind their signatures.

Stop dashboard/training before running. Default is read-only. Metrics are preserved.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools.repair_migrated_sugar import inside, signature, digest
from pipeline_tools.dataset_registry import has_evaluation


def repair(source, target, apply=False):
    source, target = Path(source).resolve(), Path(target).resolve()
    if not source.is_dir() or not target.is_dir():
        raise ValueError('Sumber/tujuan tidak ada.')
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Sumber dan tujuan harus terpisah.')
    reports = sorted(source.rglob('evaluation/metrics.json'))
    changed = skipped = 0
    for index, original_path in enumerate(reports, 1):
        relative = original_path.relative_to(source)
        old_path, new_path = inside(source, relative), inside(target, relative)
        model = relative.parent.parent
        if not has_evaluation(old_path.parent.parent):
            print('LEWATI: evaluasi cadangan tidak valid:', model, flush=True)
            skipped += 1
            continue
        original, current = old_path.read_bytes(), new_path.read_bytes()
        data, destination = json.loads(original), json.loads(current)
        old_checkpoint = inside(source, model / data['checkpoint'])
        new_checkpoint = inside(target, model / data['checkpoint'])
        # Permit reruns only when signature is the sole semantic difference.
        comparison = dict(destination)
        comparison['checkpoint_signature'] = data['checkpoint_signature']
        if comparison != data:
            raise ValueError('Laporan memiliki perbedaan selain signature: ' + str(relative))
        old_stat, new_stat = signature(old_checkpoint), signature(new_checkpoint)
        if data['checkpoint_signature'] != old_stat:
            raise ValueError('Checkpoint cadangan berubah: ' + str(model))
        if old_stat['size'] <= 0 or old_stat['size'] != new_stat['size']:
            raise ValueError('Ukuran checkpoint berbeda/kosong: ' + str(model))
        print(f'[{index}/{len(reports)}] SHA-256: {model}', flush=True)
        if digest(old_checkpoint) != digest(new_checkpoint):
            raise ValueError('ISI CHECKPOINT BERBEDA: ' + str(model))
        if signature(old_checkpoint) != old_stat or signature(new_checkpoint) != new_stat:
            raise ValueError('Checkpoint berubah selama pemeriksaan.')
        if destination['checkpoint_signature'] == new_stat:
            print('SUDAH VALID:', model, flush=True)
            continue
        if old_path.read_bytes() != original or new_path.read_bytes() != current:
            raise ValueError('Laporan berubah selama pemeriksaan.')
        if not apply:
            print('DRY RUN: hanya checkpoint_signature perlu diperbarui.', flush=True)
            changed += 1
            continue
        backup = Path(tempfile.mkdtemp(prefix='backup-repair-evaluation-', dir=target.parent))
        saved = backup / relative
        saved.parent.mkdir(parents=True)
        with saved.open('wb') as stream:
            stream.write(current)
            stream.flush()
            os.fsync(stream.fileno())
        if saved.read_bytes() != current:
            raise OSError('Verifikasi cadangan gagal.')
        updated = dict(destination)
        updated['checkpoint_signature'] = new_stat
        fd, staging = tempfile.mkstemp(prefix='.evaluation-repair-', dir=new_path.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(updated, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if (new_path.read_bytes() != current or old_path.read_bytes() != original
                or signature(new_checkpoint) != new_stat or signature(old_checkpoint) != old_stat):
            raise ValueError('Data berubah sebelum penerapan. File sementara dipertahankan.')
        os.replace(staging, new_path)
        if not has_evaluation(new_path.parent.parent):
            raise ValueError('Validasi setelah perbaikan gagal; cadangan: ' + str(saved))
        changed += 1
        print('BERHASIL; cadangan:', saved, flush=True)
    print(f'SELESAI: laporan={len(reports)}, ' +
          f'{"diperbaiki" if apply else "perlu perbaikan"}={changed}, dilewati={skipped}')
    print('Nilai metrik dan isi checkpoint tidak diubah. Status resume belum diaudit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        repair(args.source, args.target, args.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'STOP: ' + str(error) + '\n')
