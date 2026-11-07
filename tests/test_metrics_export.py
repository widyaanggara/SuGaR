"""Excel export contract: complete scope, paired models, typed numbers and safe text."""
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from web_dashboard.metrics_export import build_excel, latest_metrics, image_counts
from web_dashboard.server import DashboardHandler

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def sample_rows():
    rows = []
    for method, psnr in [('3DGS', 25), ('SuGaR', 23), ('3DGS_scaffold_anchor', 25.1), ('SuGaR_scaffold_anchor', 23.2)]:
        base = {'Kelas': 'Dewa', 'Model ID': 'model_02', 'Nama Model': '=unsafe & <name>',
                'Metode': method, 'Status': 'COMPLETED'}
        rows += [dict(base, Timestamp='2026-11-06 10:00:00', **{'Peak VRAM (GB)': '4.0', 'Total Waktu (Menit)': '20'}),
                 dict(base, Timestamp='2026-11-06 11:00:00', **{'PSNR (dB)': str(psnr), 'SSIM': '0.8', 'LPIPS': '0.2', 'Peak VRAM (GB)': '99', 'Total Waktu (Menit)': '99'})]
    rows.append(dict(rows[-1], Status='FAILED', **{'PSNR (dB)': '999'}))
    rows.append({'Kelas': 'Dewi', 'Model ID': 'model_01', 'Nama Model': 'Saraswati',
                 'Metode': '3DGS', 'Status': 'COMPLETED', 'PSNR (dB)': '0'})
    return rows


class ExcelExportTests(unittest.TestCase):
    def test_http_excel_download_contains_both_experiments(self):
        handler = object.__new__(DashboardHandler)
        handler.path = '/api/metrics.xlsx?experiment=baseline&method=3DGS'
        handler.server = SimpleNamespace(project_root=Path('.'))
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        with patch('web_dashboard.server.get_metrics_table', return_value=sample_rows()):
            handler.do_GET()
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_any_call('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        with ZipFile(io.BytesIO(handler.wfile.getvalue())) as archive:
            ablation = ET.fromstring(archive.read('xl/worksheets/sheet3.xml'))
            self.assertEqual(len(ablation.findall('s:sheetData/s:row', NS)), 6)

    def test_merge_retains_training_resources_and_sorts_history(self):
        result = latest_metrics(list(reversed(sample_rows())))
        self.assertEqual(len(result), 5)
        row = next(r for r in result if r['Metode'] == 'SuGaR')
        self.assertEqual(row['Peak VRAM (GB)'], '4.0')
        self.assertEqual(row['Total Waktu (Menit)'], '20')
        self.assertEqual(row['PSNR (dB)'], '23')

    def test_excel_package_scope_numbers_pairing_and_safe_text(self):
        with tempfile.TemporaryDirectory() as root:
            payload = build_excel(sample_rows(), root)
        with ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                ET.fromstring(archive.read(name))
            workbook = ET.fromstring(archive.read('xl/workbook.xml'))
            self.assertEqual([x.attrib['name'] for x in workbook.findall('s:sheets/s:sheet', NS)],
                             ['Perbandingan', 'Baseline', 'Studi ablasi', 'Panduan'])
            baseline = ET.fromstring(archive.read('xl/worksheets/sheet2.xml'))
            ablation = ET.fromstring(archive.read('xl/worksheets/sheet3.xml'))
            self.assertEqual(len(baseline.findall('s:sheetData/s:row', NS)), 7)
            self.assertEqual(len(ablation.findall('s:sheetData/s:row', NS)), 6)
            cells = {c.attrib['r']: c for c in baseline.findall('.//s:c', NS)}
            self.assertEqual(cells['I5'].find('s:v', NS).text, '25.0')
            self.assertEqual(cells['I6'].find('s:v', NS).text, '23.0')
            self.assertEqual(cells['C5'].find('s:is/s:t', NS).text, '=unsafe & <name>')
            self.assertEqual(cells['C5'].attrib['t'], 'inlineStr')
            self.assertEqual(cells['D5'].find('s:is/s:t', NS).text, '3DGS')
            self.assertEqual(cells['D6'].find('s:is/s:t', NS).text, 'SuGaR')
            self.assertEqual(cells['C5'].attrib['s'], cells['C6'].attrib['s'])
            self.assertNotEqual(cells['C6'].attrib['s'], cells['C7'].attrib['s'])
            self.assertEqual(cells['I7'].find('s:v', NS).text, '0.0')
            self.assertFalse(baseline.findall('.//s:f', NS))
            self.assertEqual(baseline.find('s:autoFilter', NS).attrib['ref'], 'A4:T7')
            self.assertEqual(baseline.find('.//s:pane', NS).attrib['topLeftCell'], 'D5')
            delta = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
            differences = {c.attrib['r']: c for c in delta.findall('.//s:c', NS)}
            self.assertAlmostEqual(float(differences['E5'].find('s:v', NS).text), 0.1)
            self.assertAlmostEqual(float(differences['E6'].find('s:v', NS).text), 0.2)

    def test_counts_are_from_matching_run_and_legacy_missing_stays_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'outputs/3dgs/Dewa/model_02/evaluation/metrics.json'
            path.parent.mkdir(parents=True)
            report = dict(psnr=25, ssim=0.8, lpips=0.2, registered_images=100, train_images=87, test_images=13)
            path.write_text(json.dumps(report))
            row = sample_rows()[1]
            self.assertEqual(image_counts(root, 'Dewa', 'model_02', 'baseline', row), [100, 87, 13])
            report = dict(psnr=25, ssim=0.8, lpips=0.2, per_view=[{}, {}])
            path.write_text(json.dumps(report))
            self.assertEqual(image_counts(root, 'Dewa', 'model_02', 'baseline', row), [None, None, 2])
            row['PSNR (dB)'] = '26'
            self.assertEqual(image_counts(root, 'Dewa', 'model_02', 'baseline', row), [None]*3)

    def test_empty_export_is_a_valid_workbook(self):
        with ZipFile(io.BytesIO(build_excel([], '.'))) as archive:
            self.assertIsNone(archive.testzip())
            for name in archive.namelist():
                ET.fromstring(archive.read(name))

    def test_legacy_camera_counts_for_both_methods_and_experiments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for experiment, total in [('baseline', 17), ('scaffold_anchor', 25)]:
                base = root / 'outputs'
                suffix = ''
                if experiment != 'baseline':
                    base /= 'ablations/scaffold_anchor'
                    suffix = '_scaffold_anchor'
                names = [f'IMG_{i:03d}' for i in range(total)]
                path = base / '3dgs/Dewa/model_02/cameras.json'
                path.parent.mkdir(parents=True)
                # File order differs from evaluation order; duplicates match loader semantics.
                path.write_text(json.dumps([{'img_name': name} for name in reversed(names)] + [{'img_name': names[0]}]))
                for method in ('3DGS', 'SuGaR'):
                    report_path = base / method.lower() / 'Dewa/model_02/evaluation/metrics.json'
                    report_path.parent.mkdir(parents=True)
                    report_path.write_text(json.dumps(dict(psnr=25, ssim=0.8, lpips=0.2,
                        eval_split_interval=8, per_view=[{'image': name} for name in names[::8]])))
                    row = dict(sample_rows()[1], Metode=method + suffix)
                    self.assertEqual(image_counts(root, 'Dewa', 'model_02', experiment, row),
                                     [total, total - len(names[::8]), len(names[::8])])
                    with ZipFile(io.BytesIO(build_excel([row], root))) as archive:
                        sheet = ET.fromstring(archive.read('xl/worksheets/sheet' + ('2' if experiment == 'baseline' else '3') + '.xml'))
                        cells = {c.attrib['r']: c for c in sheet.findall('.//s:c', NS)}
                        self.assertEqual(int(cells['F5'].find('s:v', NS).text), total)
                        self.assertEqual(int(cells['G5'].find('s:v', NS).text), total - len(names[::8]))

    def test_legacy_counts_reject_mismatched_or_malformed_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'outputs/3dgs/Dewa/model_02'
            report_path = base / 'evaluation/metrics.json'
            report_path.parent.mkdir(parents=True)
            cameras = [{'img_name': f'IMG_{i:03d}'} for i in range(17)]
            (base / 'cameras.json').write_text(json.dumps(cameras))
            report = dict(psnr=25, ssim=0.8, lpips=0.2, eval_split_interval=8,
                          per_view=[{'image': f'IMG_{i:03d}'} for i in (0, 8, 16)])
            for change in [dict(eval_split_interval=0), dict(eval_split_interval=True),
                           dict(eval_split_interval=None), dict(per_view=[{'image': 'OTHER'}]),
                           dict(per_view=[{'image': 'IMG_000'}]*3)]:
                changed = dict(report, **change)
                report_path.write_text(json.dumps(changed))
                self.assertEqual(image_counts(root, 'Dewa', 'model_02', 'baseline', sample_rows()[1]),
                                 [None, None, len(changed['per_view'])])
            report_path.write_text(json.dumps(report))
            for malformed in ['{invalid', '{}', '[{}]', '[null]', '[]']:
                (base / 'cameras.json').write_text(malformed)
                self.assertEqual(image_counts(root, 'Dewa', 'model_02', 'baseline', sample_rows()[1]),
                                 [None, None, 3])


if __name__ == '__main__':
    unittest.main()
