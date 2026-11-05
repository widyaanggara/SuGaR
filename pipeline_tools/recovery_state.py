"""Atomic, configuration-bound recovery metadata. No GPU imports on the server."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from pipeline_tools.dataset_registry import file_signature, media_files
from pipeline_tools.run_settings import colmap_signature


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def scene_identity(scene):
    scene = Path(scene)
    return {'scene': str(scene.resolve()), 'colmap': colmap_signature(scene),
            'images': {path.name: file_signature(path) for path in media_files(scene / 'images')}}


def code_identity(root, paths):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in paths if (Path(root) / name).is_file()}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def configure_recovery(path, context, resume=True):
    """Fresh runs get their own generation; retries cannot mix two runs."""
    path = Path(path)
    if resume:
        try:
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous['context'] == context and isinstance(previous['generation'], str):
                return previous
        except (OSError, ValueError, KeyError, TypeError):
            pass
    identity = {'context': context, 'generation': uuid.uuid4().hex}
    atomic_json(path, identity)
    return identity


class CheckpointStore:
    """Two rotating slots per configuration, with an untouched previous slot.

    Heavy serialization/deserialization is supplied by the training process.
    Metadata is only a discovery hint; the envelope is validated after loading.
    """
    def __init__(self, directory, identity, log=print):
        self.identity = identity
        self.key = fingerprint(identity)
        self.directory = Path(directory) / self.key
        self.log = log
        self.current = None

    def candidates(self):
        candidates = []
        for slot in ('a', 'b'):
            path = self.directory / f'{slot}.pt'
            try:
                meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                if (meta['fingerprint'] == self.key and path.is_file()
                        and isinstance(meta['iteration'], int) and meta['iteration'] >= 0):
                    candidates.append((meta['iteration'], path))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(candidates, reverse=True)

    def load(self, loader, validate):
        candidates = self.candidates()
        if not candidates and self.directory.exists() and any(self.directory.glob('*.pt')):
            raise ValueError('Checkpoint ada tetapi metadata tidak dapat diverifikasi. Tidak memulai ulang diam-diam.')
        for iteration, path in candidates:
            try:
                meta = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
                if meta['signature'] != file_signature(path):
                    raise ValueError('Ukuran/waktu checkpoint tidak cocok dengan metadata atomik.')
                envelope = loader(path)
                if (envelope['fingerprint'] != self.key or envelope['iteration'] != iteration
                        or envelope['schema'] != 1):
                    raise ValueError('Metadata checkpoint tidak cocok.')
                validate(envelope['state'], iteration)
                self.log(f'[RESUME] Checkpoint valid: {path}; iterasi {iteration}.')
                self.current = path
                return envelope['state'], iteration
            except Exception as exc:
                # CUDA OOM and resource exhaustion must not be disguised as a corrupt file.
                if isinstance(exc, MemoryError) or 'out of memory' in str(exc).lower():
                    raise
                self.log(f'[RESUME] Checkpoint tidak dapat dipakai: {path.name}: {exc}; mencoba cadangan.')
        if candidates:
            raise ValueError('Tidak ada checkpoint valid untuk konfigurasi ini. Periksa log; training tidak dimulai dari nol secara diam-diam.')
        return None, 0

    def save(self, state, iteration, serializer):
        self.directory.mkdir(parents=True, exist_ok=True)
        candidates = self.candidates()
        newest = self.current.stem if self.current is not None else candidates[0][1].stem if candidates else 'b'
        path = self.directory / ('b.pt' if newest == 'a' else 'a.pt')
        temporary = path.with_suffix('.pt.tmp')
        try:
            with temporary.open('wb') as stream:
                serializer({'schema': 1, 'fingerprint': self.key, 'iteration': int(iteration), 'state': state}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            atomic_json(path.with_suffix('.json'), {'fingerprint': self.key, 'iteration': int(iteration),
                'signature': file_signature(path), 'saved_at': time.time(), 'identity': self.identity})
            self.current = path
        finally:
            temporary.unlink(missing_ok=True)


class StageJournal:
    """Reuse only completed stages with matching configuration and input signatures."""
    def __init__(self, directory, identity, enabled=True, log=print):
        self.directory = Path(directory)
        self.identity = identity
        self.enabled = enabled
        self.log = log

    def get(self, stage, inputs):
        if not self.enabled:
            return None
        try:
            record = json.loads((self.directory / f'{stage}.json').read_text(encoding='utf-8'))
            if record['identity'] != self.identity or record['inputs'] != inputs:
                return None
            path = Path(record['output'])
            if (record['signature'] == file_signature(path) and path.stat().st_size and
                    all(file_signature(Path(name)) == signature for name, signature in record.get('dependencies', {}).items())):
                self.log(f'[RESUME] Tahap {stage} sudah lengkap; memakai {path}.')
                return str(path)
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def complete(self, stage, inputs, output):
        path = Path(output).resolve()
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Hasil tahap {stage} kosong atau tidak ditemukan.')
        dependencies = {}
        if path.suffix.lower() == '.obj':
            with path.open(encoding='utf-8', errors='replace') as stream:
                materials = [line.strip()[7:].strip() for line in stream if line.startswith('mtllib ')]
            for name in materials:
                material = (path.parent / name).resolve()
                dependencies[str(material)] = file_signature(material)
                with material.open(encoding='utf-8', errors='replace') as stream:
                    for line in stream:
                        if line.startswith('map_Kd '):
                            texture = (material.parent / line.strip()[7:].strip()).resolve()
                            dependencies[str(texture)] = file_signature(texture)
            if any(signature['size'] == 0 for signature in dependencies.values()):
                raise ValueError('Material atau tekstur OBJ kosong.')
        atomic_json(self.directory / f'{stage}.json', {'identity': self.identity, 'inputs': inputs,
            'output': str(path), 'signature': file_signature(path), 'dependencies': dependencies})
