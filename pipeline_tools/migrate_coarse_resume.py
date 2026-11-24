"""Migrate an interrupted coarse SuGaR run, retaining its old checkpoints.

Default is read-only. --apply requires the original training Python environment.
Only use on trusted, local checkpoints. Stop all output writers first.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools.repair_migrated_sugar import inside, signature, digest
from pipeline_tools.recovery_state import fingerprint, scene_identity, code_identity


def write_json(path, value):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def migrate(source, target, model, root, apply=False, runtime=None):
    source, target, root = (Path(p).resolve() for p in (source, target, root))
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Sumber dan tujuan harus terpisah.')
    model = Path(model)
    if len(model.parts) < 3 or model.parts[-3] != 'sugar':
        raise ValueError('Model harus menunjuk folder sugar/Kelas/Model.')
    old_out, out = inside(source, model), inside(target, model)
    identity_path = out / 'recovery/identity.json'
    original = (old_out / 'recovery/identity.json').read_bytes()
    if identity_path.read_bytes() != original:
        raise ValueError('Identitas sudah berbeda; jangan migrasikan ulang tanpa audit.')
    identity = json.loads(original)
    context = identity['context']
    if context.get('reused_coarse') is not None:
        raise ValueError('Alat ini khusus coarse pertama, bukan reused_coarse.')
    if (out / 'pipeline_result.json').exists() or any((out / 'recovery/stages').glob('*/*.json')):
        raise ValueError('Sudah ada hasil/catatan tahap. Perlu audit tersendiri.')
    if scene_identity(context['scene']['scene']) != context['scene']:
        raise ValueError('Dataset berubah.')
    if code_identity(root, context['code']) != context['code']:
        raise ValueError('Kode training berubah.')
    baseline_rel = model.parent.parent.parent / '3dgs' / model.parts[-2] / model.parts[-1]
    baseline = baseline_rel / 'point_cloud/iteration_7000/point_cloud.ply'
    old_baseline, new_baseline = inside(source, baseline), inside(target, baseline)
    if signature(old_baseline) != context['baseline']:
        raise ValueError('Baseline cadangan tidak cocok dengan identitas asli.')
    if code_identity(inside(target, baseline_rel), context['cameras']) != context['cameras']:
        raise ValueError('Kamera baseline berubah.')
    verified = []

    def verify(old, new, expected):
        old_stat, new_stat = signature(old), signature(new)
        if old_stat != expected or old_stat['size'] != new_stat['size'] or not new_stat['size']:
            raise ValueError('Metadata cadangan atau ukuran file tidak cocok: ' + str(new))
        print('SHA-256:', new, flush=True)
        if digest(old) != digest(new):
            raise ValueError('Isi file berbeda: ' + str(new))
        if signature(old) != old_stat or signature(new) != new_stat:
            raise ValueError('File berubah selama pemeriksaan.')
        verified.extend(((old, old_stat), (new, new_stat)))
        return new_stat

    new_baseline_stat = verify(old_baseline, new_baseline, context['baseline'])
    new_pipeline = copy.deepcopy(identity)
    new_pipeline['context']['baseline'] = new_baseline_stat
    groups = list((old_out / 'recovery/checkpoints/coarse').glob('*'))
    groups = [p for p in groups if p.is_dir() and any(p.glob('[ab].json'))]
    if len(groups) != 1:
        raise ValueError('Harus tepat satu kelompok checkpoint coarse.')
    group = groups[0]
    records = []
    destination_key = None
    for old_meta in sorted(group.glob('[ab].json')):
        new_meta = inside(target, old_meta.relative_to(source))
        raw = old_meta.read_bytes()
        if new_meta.read_bytes() != raw:
            raise ValueError('Metadata slot berbeda dari cadangan.')
        record = json.loads(raw)
        slot_identity = record['identity']
        if (slot_identity.get('pipeline') != identity or slot_identity.get('stage') != 'coarse'
                or slot_identity.get('inputs') != {} or fingerprint(slot_identity) != group.name
                or record['fingerprint'] != group.name):
            raise ValueError('Identitas/fingerprint slot tidak sesuai.')
        verify(old_meta.with_suffix('.pt'), new_meta.with_suffix('.pt'), record['signature'])
        changed_identity = copy.deepcopy(slot_identity)
        changed_identity['pipeline'] = new_pipeline
        key = fingerprint(changed_identity)
        if destination_key is not None and key != destination_key:
            raise ValueError('Identitas antar slot berbeda.')
        destination_key = key
        records.append((new_meta, raw, record, changed_identity))
    if not records:
        raise ValueError('Tidak ada checkpoint.')
    print('Checkpoint terverifikasi:', [r[2]['iteration'] for r in records], flush=True)
    if not apply:
        print('DRY RUN: belum ada perubahan. Apply membuat salinan checkpoint baru, bukan menimpa slot lama.')
        return
    if runtime is None:
        import torch
        import numpy
        runtime = (torch, numpy.__version__)
    torch, numpy_version = runtime
    for _, _, record, _ in records:
        if str(torch.__version__) != record['identity']['torch'] or numpy_version != record['identity']['numpy']:
            raise ValueError('Gunakan environment training dengan versi Torch/NumPy yang sama.')
    parent = out / 'recovery/checkpoints/coarse'
    destination = parent / destination_key
    if destination.exists():
        raise ValueError('Folder fingerprint tujuan sudah ada; hentikan untuk audit.')
    required = sum(meta.with_suffix('.pt').stat().st_size for meta, *_ in records)
    if shutil.disk_usage(parent).free < required * 1.2 + 1024**3:
        raise ValueError('Ruang F tidak cukup untuk salinan checkpoint baru.')
    staging = Path(tempfile.mkdtemp(prefix='.migration-coarse-', dir=parent))
    for meta, raw, record, changed_identity in records:
        # Trusted user-owned checkpoints, checksum verified against untouched source.
        envelope = torch.load(meta.with_suffix('.pt'), map_location='cpu', weights_only=False)
        if (envelope.get('schema') != 1 or envelope.get('fingerprint') != record['fingerprint']
                or envelope.get('iteration') != record['iteration']):
            raise ValueError('Envelope checkpoint tidak sesuai metadata.')
        if not isinstance(envelope.get('state'), dict) or not all(k in envelope['state'] for k in
                ('model', 'optimizer', 'rng', 'training', 'camera_order', 'camera_offset', 'buffers')):
            raise ValueError('State recovery tidak lengkap.')
        # All tensors, optimizer, RNG and iteration stay in the same state object.
        envelope['fingerprint'] = destination_key
        result = staging / meta.with_suffix('.pt').name
        with result.open('wb') as stream:
            torch.save(envelope, stream)
            stream.flush()
            os.fsync(stream.fileno())
        del envelope
        check = torch.load(result, map_location='cpu', weights_only=False)
        if check.get('fingerprint') != destination_key or check.get('iteration') != record['iteration']:
            raise ValueError('Pembacaan ulang checkpoint baru gagal.')
        del check
        updated = copy.deepcopy(record)
        updated.update(identity=changed_identity, fingerprint=destination_key, signature=signature(result))
        write_json(staging / meta.name, updated)
    if identity_path.read_bytes() != original or any(signature(p) != s for p, s in verified):
        raise ValueError('Data berubah; identitas belum diganti. Salinan sementara dipertahankan.')
    if any(meta.read_bytes() != raw for meta, raw, *_ in records):
        raise ValueError('Metadata slot berubah; identitas belum diganti.')
    backup = Path(tempfile.mkdtemp(prefix='backup-migrate-coarse-', dir=target.parent))
    with (backup / 'identity.before.json').open('wb') as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    if (backup / 'identity.before.json').read_bytes() != original:
        raise ValueError('Cadangan identitas gagal diverifikasi.')
    write_json(backup / 'migration.json', {'model': str(model), 'old': group.name, 'new': destination_key})
    staging.rename(destination)
    for meta, _, _, _ in records:
        record = json.loads((destination / meta.name).read_text())
        if record['signature'] != signature((destination / meta.name).with_suffix('.pt')):
            raise ValueError('Metadata checkpoint berubah setelah rename; identitas belum diganti.')
    pending = identity_path.with_name('identity.migration-pending.json')
    if pending.exists():
        raise ValueError('File identitas sementara sudah ada; perlu audit.')
    write_json(pending, new_pipeline)
    os.replace(pending, identity_path)
    print('BERHASIL: identitas coarse dimigrasikan; checkpoint lama tetap ada.')
    print('Cadangan identitas:', backup)
    print('Iterasi tertinggi:', max(r[2]['iteration'] for r in records))
    print('Belum menjalankan training. Konfigurasi antrean dan baseline training_settings masih harus cocok.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='outputs_backup_lama')
    parser.add_argument('--target', default='outputs')
    parser.add_argument('--root', default='.')
    parser.add_argument('--model', default='ablations/scaffold_anchor/sugar/Punakawan/model_02')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        migrate(args.source, args.target, args.model, args.root, args.apply)
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as error:
        parser.exit(1, 'STOP: ' + str(error) + '\n')
