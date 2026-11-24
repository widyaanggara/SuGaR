"""Repair Scaffold receipts and 3DGS initialization for one migrated model.

Read-only by default. Stop dashboard and training. Checkpoints are never written.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_tools.repair_migrated_sugar import inside, signature, digest
from pipeline_tools.recovery_state import scene_identity, code_identity, fingerprint
from pipeline_tools.run_settings import colmap_signature, read_training_config


def repair(root, source, target, apply=False):
    root, source, target = (Path(p).resolve() for p in (root, source, target))
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Sumber dan tujuan harus terpisah.')
    model = Path('ablations/scaffold_anchor/3dgs/Punakawan/model_02')
    settings_rel = model / 'training_settings.json'
    originals, changes, verified = {}, {}, []

    def receipt(relative):
        old, new = inside(source, relative), inside(target, relative)
        raw = old.read_bytes()
        if new.read_bytes() != raw:
            raise ValueError('Catatan sudah berbeda; perlu audit: ' + str(relative))
        originals[relative] = raw
        return json.loads(raw)

    def reference(path):
        # Preserve lexical original outputs path; resolving first would lose it.
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Path catatan tidak aman: ' + str(path))
        for base in (root / 'outputs', source, target):
            try:
                return path.relative_to(base)
            except ValueError:
                pass
        raise ValueError('Path di luar output: ' + str(path))

    def verify(relative, expected):
        old, new = inside(source, relative), inside(target, relative)
        before, after = signature(old), signature(new)
        if before != expected or before['size'] != after['size'] or not before['size']:
            raise ValueError('Ukuran/metadata cadangan tidak cocok: ' + str(relative))
        print('SHA-256:', relative, flush=True)
        if digest(old) != digest(new):
            raise ValueError('Isi file berbeda: ' + str(relative))
        if signature(old) != before or signature(new) != after:
            raise ValueError('File berubah selama pemeriksaan.')
        verified.extend(((old, before), (new, after)))
        return after

    settings = receipt(settings_rel)
    initial = reference(settings['initialization']['path'])
    scaffold = Path('ablations/scaffold_anchor/scaffold/Punakawan/model_02/runs')
    if initial.parent.parent != scaffold or initial.name != 'initial_points.ply':
        raise ValueError('Inisialisasi bukan run Scaffold yang diharapkan.')
    run = initial.parent
    train_rel, convert_rel = run / 'stages/training.json', run / 'stages/conversion.json'
    train, convert = receipt(train_rel), receipt(convert_rel)
    identity = train['identity']
    if convert['identity'] != identity or fingerprint(identity) != run.name or train['inputs'] != {}:
        raise ValueError('Identitas run Scaffold tidak konsisten.')
    if train.get('dependencies') or convert.get('dependencies'):
        raise ValueError('Dependensi tambahan perlu audit.')
    scene = identity['scene']['scene']
    if scene_identity(scene) != identity['scene']:
        raise ValueError('Dataset Scaffold berubah.')
    repo = Path(identity['repo'])
    files = [str(p.relative_to(repo)) for p in repo.rglob('*.py') if '.git' not in p.parts]
    if (code_identity(repo, files) != identity['code'] or
            code_identity(root, ['pipeline_tools/scaffold_init.py', 'pipeline_tools/scaffold_train.py']) != identity['adapter']):
        raise ValueError('Kode Scaffold/adapter berubah.')
    anchor = reference(train['output'])
    if anchor != run / f"point_cloud/iteration_{identity['iterations']}/point_cloud.ply":
        raise ValueError('Path checkpoint Scaffold tidak sesuai.')
    if reference(convert['output']) != initial or convert['inputs'] != {'anchor_ply': train['signature']}:
        raise ValueError('Catatan konversi tidak sesuai training.')
    if settings['initialization']['signature'] != convert['signature']:
        raise ValueError('3DGS tidak mengacu pada hasil konversi yang sama.')
    if settings['colmap_signature'] != colmap_signature(scene):
        raise ValueError('COLMAP baseline berubah.')
    cfg = read_training_config(inside(target, model))
    if cfg.get('eval') is not True or cfg.get('white_background', False) != settings.get('white_background', False):
        raise ValueError('Split/background baseline tidak cocok.')
    anchor_sig = verify(anchor, train['signature'])
    initial_sig = verify(initial, convert['signature'])
    new_train, new_convert, new_settings = map(copy.deepcopy, (train, convert, settings))
    new_train.update(output=str(inside(target, anchor)), signature=anchor_sig)
    new_convert.update(output=str(inside(target, initial)), signature=initial_sig, inputs={'anchor_ply': anchor_sig})
    new_settings['initialization'] = {'path': str(inside(target, initial)), 'signature': initial_sig}
    changes.update({train_rel: new_train, convert_rel: new_convert, settings_rel: new_settings})
    print('Lolos: checkpoint Scaffold dan initial_points identik; tiga catatan perlu disesuaikan.')
    if not apply:
        print('DRY RUN: tidak ada file yang diubah.')
        return
    if any(signature(p) != s for p, s in verified):
        raise ValueError('File berubah sebelum penerapan.')
    for relative, raw in originals.items():
        if inside(source, relative).read_bytes() != raw or inside(target, relative).read_bytes() != raw:
            raise ValueError('Catatan berubah sebelum penerapan.')
    backup = Path(tempfile.mkdtemp(prefix='backup-scaffold-migration-', dir=target.parent))
    for relative, raw in originals.items():
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open('wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if saved.read_bytes() != raw:
            raise ValueError('Verifikasi cadangan gagal.')
    print('Cadangan seluruh catatan:', backup, flush=True)
    for relative, value in changes.items():
        path = inside(target, relative)
        fd, staging = tempfile.mkstemp(prefix='.scaffold-migration-', dir=path.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    print('BERHASIL: tiga catatan diperbarui; isi checkpoint, initial_points dan cadangan asli tidak diubah.')
    print('Belum menjalankan training. Periksa baseline dan rencana antrean sebelum resume.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--source', default='outputs_backup_lama')
    parser.add_argument('--target', default='outputs')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        repair(args.root, args.source, args.target, args.apply)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, 'STOP: ' + str(error) + '\n')
