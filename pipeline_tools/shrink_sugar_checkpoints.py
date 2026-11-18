"""Shrink finished SuGaR refined checkpoints by dropping the unused Adam optimizer state.

Rendering, mesh export and evaluation only read checkpoint['state_dict'], which is saved again
unchanged, so results stay identical. Older runs stored the optimizer state too (about 2x the model).
Records that point at a shrunk file (evaluation/metrics.json, the stage journal) get the new file
signature, but only when they matched the file before shrinking.

    python pipeline_tools/shrink_sugar_checkpoints.py           # list what would be shrunk
    python pipeline_tools/shrink_sugar_checkpoints.py --apply   # shrink
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_tools.dataset_registry import file_signature

OUTPUT_ROOTS = ('outputs/sugar', 'outputs/ablations/scaffold_anchor/sugar', 'outputs/quality/sugar')


def sugar_outputs(root):
    for base in OUTPUT_ROOTS:
        yield from sorted(path.parent for path in (Path(root) / base).rglob('pipeline_result.json'))


def refined_checkpoints(output):
    return sorted(Path(output).glob('refined/*/*.pt'))


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def shrink(checkpoint, output, load, save):
    """Drop the optimizer state; returns True when the file was rewritten."""
    checkpoint = Path(checkpoint).resolve()
    before = file_signature(checkpoint)
    data = load(checkpoint)
    if 'optimizer_state_dict' not in data:
        return False
    if 'state_dict' not in data:
        raise ValueError(f'{checkpoint}: tidak ada state_dict; file tidak diubah.')
    del data['optimizer_state_dict']
    save(data, checkpoint)
    after = file_signature(checkpoint)
    metrics = Path(output) / 'evaluation/metrics.json'
    if metrics.is_file():
        report = json.loads(metrics.read_text(encoding='utf-8'))
        if ((Path(output) / report.get('checkpoint', '')).resolve() == checkpoint
                and report.get('checkpoint_signature') == before):
            report.update(checkpoint_signature=after, checkpoint_optimizer_state_removed=True)
            _write_json(metrics, report)
    for record_path in Path(output).glob('recovery/stages/*/refine.json'):
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if Path(record.get('output', '')).resolve() == checkpoint and record.get('signature') == before:
            record['signature'] = after
            _write_json(record_path, record)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true', help='Tulis ulang file; tanpa ini hanya menampilkan daftar.')
    args = parser.parse_args()
    import torch
    from sugar_utils.training_recovery import atomic_torch_save
    load = lambda path: torch.load(str(path), map_location='cpu', weights_only=False)
    total_before = total_after = 0
    for output in sugar_outputs(ROOT):
        for checkpoint in refined_checkpoints(output):
            size = checkpoint.stat().st_size
            if not args.apply:
                print(f'{size / 1e9:6.2f} GB  {checkpoint.relative_to(ROOT)}')
                continue
            changed = shrink(checkpoint, output, load, atomic_torch_save)
            new_size = checkpoint.stat().st_size
            total_before, total_after = total_before + size, total_after + new_size
            print(f'{size / 1e9:6.2f} -> {new_size / 1e9:6.2f} GB  {checkpoint.relative_to(ROOT)}'
                  + ('' if changed else '  (sudah ringkas)'), flush=True)
    if args.apply:
        print(f'Total: {total_before / 1e9:.2f} -> {total_after / 1e9:.2f} GB')
    else:
        print('Tambahkan --apply untuk mengecilkan. Jangan jalankan saat SuGaR sedang training/evaluasi.')


if __name__ == '__main__':
    main()
