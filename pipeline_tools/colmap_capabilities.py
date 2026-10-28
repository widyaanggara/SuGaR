"""Inspect the COLMAP build without importing GPU libraries or starting work."""
import re
import subprocess


def inspect_colmap(executable='colmap'):
    try:
        result = subprocess.run([executable, '-h'], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                errors='replace', timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'cuda': False, 'version': '', 'message': str(exc)}
    output = result.stdout or ''
    cuda = bool(re.search(r'\bwith CUDA\b', output, re.IGNORECASE))
    available = 'COLMAP' in output.upper()
    header = '\n'.join(output.splitlines()[:4]).strip()
    return {'available': available, 'cuda': cuda, 'version': header,
            'message': ('Build COLMAP mendukung CUDA; perangkat akan diuji saat proses dimulai.' if cuda else
                        'Dukungan CUDA tidak terdeteksi pada COLMAP. Mode otomatis memakai CPU.')}


def resolve_device(device, executable='colmap'):
    if device == 'cpu':
        return False
    info = inspect_colmap(executable)
    if not info['available']:
        raise RuntimeError('COLMAP tidak dapat diperiksa: ' + info['message'])
    if device == 'cuda' and not info['cuda']:
        raise RuntimeError('GPU COLMAP dipilih tetapi build CUDA tidak terdeteksi. '
                           'Pasang COLMAP dengan CUDA, atau pilih Otomatis / CPU di dashboard.')
    return info['cuda']
