"""One queue job and combined log per statue; evaluate methods sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from web_dashboard.server import build_command
from pipeline_tools.geometry_config import geometry_result_current


def run(job, root=ROOT):
    methods = job.get('evaluation_methods')
    if (not isinstance(methods, list) or not methods or
            any(not isinstance(method, str) or method not in ('3DGS', 'SuGaR', '2DGS', 'RaDe-GS')
                for method in methods) or len(set(methods)) != len(methods)):
        raise ValueError('Daftar metode geometri tidak valid.')
    identity = f"{job['class_name']}/{job['model_id']}"
    print(f'\n[TOGOG] Evaluasi Geometri: {identity}', flush=True)
    print(f"[EKSPERIMEN] {job.get('experiment', 'baseline')}", flush=True)
    print('[METODE DIPILIH] ' + ', '.join(methods), flush=True)
    results = []
    for index, method in enumerate(methods, 1):
        print(f'\n========== [{index}/{len(methods)}] {identity} | {method} ==========', flush=True)
        single = {key: value for key, value in job.items() if key != 'evaluation_methods'}
        single.update(action='evaluate', evaluation_method=method)
        try:
            if geometry_result_current(root, job['class_name'], job['model_id'], method,
                    job.get('scale_factor'), job.get('geometry_options'),
                    job.get('experiment', 'baseline'), apply_icp=job.get('apply_icp', True)):
                print(f'[LEWATI] {method}: evaluasi sudah selesai dan masih sesuai.', flush=True)
                results.append((method, True, 'DILEWATI: hasil valid sudah tersedia'))
                continue
            command = build_command(single, root)
            print('[RUN] ' + subprocess.list2cmdline(command), flush=True)
            # Inherit stdout/stderr: all child diagnostics belong to this statue's queue log.
            result = subprocess.run(command, cwd=root)
            success = result.returncode == 0
            detail = f'exit code {result.returncode}'
        except (OSError, ValueError, KeyError) as error:
            success, detail = False, f'{type(error).__name__}: {error}'
        results.append((method, success, detail))
        print(f'[HASIL METODE] {method}: {"BERHASIL" if success else "GAGAL"} ({detail})', flush=True)
        if not success and index < len(methods):
            print('[LANJUT] Metode ini gagal; evaluasi metode berikutnya tetap dijalankan.', flush=True)
    print(f'\n========== RINGKASAN GEOMETRI {identity} ==========', flush=True)
    for method, success, detail in results:
        print(f'{method}: {"BERHASIL" if success else "GAGAL"} ({detail})', flush=True)
    failures = sum(not success for _, success, _ in results)
    print(f'[SELESAI] {len(results) - failures} berhasil; {failures} gagal. '
          'Metrik metode yang berhasil tetap disimpan; metode gagal tidak memperoleh skor baru.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job_json', required=True)
    args = parser.parse_args()
    raise SystemExit(run(json.loads(args.job_json)))
