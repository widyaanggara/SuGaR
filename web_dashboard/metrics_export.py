"""Portable Excel report for the dashboard (OOXML, no server dependencies)."""
import io
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

METRICS = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)',
           'Precision @2cm (%)', 'Recall @2cm (%)', 'F1 @2cm (%)',
           'Precision @5cm (%)', 'Recall @5cm (%)', 'F1 @5cm (%)',
           'Peak VRAM (GB)', 'Total Waktu (Menit)']
METHODS = ('3DGS', 'SuGaR', '3DGS_scaffold_anchor', 'SuGaR_scaffold_anchor')


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def latest_metrics(rows):
    """Mirror dashboard merging: evaluation memory/time never replace training."""
    merged = {}
    for row in sorted(rows, key=lambda row: row.get('Timestamp', '')):
        if row.get('Status') != 'COMPLETED' or row.get('Metode') not in METHODS:
            continue
        key = tuple(row.get(k, '') for k in ('Kelas', 'Model ID', 'Metode'))
        current = merged.setdefault(key, {})
        evaluated = any(number(row.get(k)) is not None for k in METRICS[:10])
        for column, value in row.items():
            if evaluated and column in METRICS[-2:]:
                continue
            if value not in ('-', '', None):
                current[column] = value
    return list(merged.values())


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def image_counts(root, cls, model, experiment, row):
    """Use run artifacts, never count the mutable dataset folder as historical data."""
    if not re.fullmatch(r'[\w-]+', cls) or not re.fullmatch(r'[\w-]+', model):
        return [None, None, None]
    base = Path(root) / 'outputs'
    if experiment == 'scaffold_anchor':
        base /= 'ablations/scaffold_anchor'
    method = row.get('Metode', '').replace('_scaffold_anchor', '').lower()
    report = read_json(base / method / cls / model / 'evaluation/metrics.json')
    if not isinstance(report, dict):
        return [None, None, None]
    # A newer artifact must not supply counts for a different exported result.
    for key, column in [('psnr', 'PSNR (dB)'), ('ssim', 'SSIM'), ('lpips', 'LPIPS')]:
        a, b = number(report.get(key)), number(row.get(column))
        if a is None or b is None or abs(a - b) > 0.00011:
            return [None, None, None]
    counts = [report.get(k) for k in ('registered_images', 'train_images', 'test_images')]
    if all(isinstance(v, int) and v >= 0 for v in counts) and counts[0] == sum(counts[1:]):
        return counts
    # Legacy evaluation recorded test views; do not guess the training population.
    views = report.get('per_view')
    return [None, None, len(views) if isinstance(views, list) else None]


def column_name(index):
    result = ''
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def xml_text(value):
    return escape(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value)))


def sheet_xml(title, note, headers, rows, groups=()):
    width = len(headers)
    end = column_name(width)
    data = [[title], [note], [], headers, *rows]
    cells = []
    previous_model, model_band = None, -1
    for r, values in enumerate(data, 1):
        if r > 4 and headers[0] == 'Kelas':
            model = tuple(values[:2])
            if model != previous_model:
                model_band += 1
                previous_model = model
        body_style = (4 + model_band % 2) if model_band >= 0 else (4 if r % 2 else 5)
        contents = []
        for c, value in enumerate(values, 1):
            style = 1 if r == 1 else 2 if r in (3, 4) else 3 if r == 2 else body_style
            if value is None:
                value = '—'
            ref = f'{column_name(c)}{r}'
            if isinstance(value, (float, int)):
                numeric_style = style + 2 if isinstance(value, float) and style in (4, 5) else style
                contents.append(f'<c r="{ref}" s="{numeric_style}"><v>{value}</v></c>')
            else:
                contents.append(f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{xml_text(value)}</t></is></c>')
        height = 42 if r in (2, 4) else 26
        if r > 4 and len(values) > 2:
            height = max(height, 16 * math.ceil(len(str(values[2] or '')) / 25))
        cells.append(f'<row r="{r}" ht="{height}" customHeight="1">{"".join(contents)}</row>')
    merges = [f'A1:{column_name(min(width, 12))}1', f'A2:{column_name(min(width, 19))}2']
    if groups:
        group_cells = []
        for start, stop, label in groups:
            group_cells.append(f'<c r="{column_name(start)}3" s="2" t="inlineStr"><is><t>{xml_text(label)}</t></is></c>')
            if stop > start:
                merges.append(f'{column_name(start)}3:{column_name(stop)}3')
        cells[2] = f'<row r="3" ht="26" customHeight="1">{"".join(group_cells)}</row>'
    cols = ''.join(f'<col min="{i}" max="{i}" width="{27 if i == 3 or headers[i-1] == "Waktu hasil terakhir" else 19 if i <= 3 else 17}" customWidth="1"/>' for i in range(1, width + 1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="A1:{end}{len(data)}"/>'
        '<sheetViews><sheetView workbookViewId="0" showGridLines="0"><pane xSplit="3" ySplit="4" topLeftCell="D5" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>'
        f'<cols>{cols}</cols><sheetData>{"".join(cells)}</sheetData>'
        f'<autoFilter ref="A4:{end}{max(4, len(data))}"/>'
        f'<mergeCells count="{len(merges)}">' + ''.join(f'<mergeCell ref="{ref}"/>' for ref in merges) + '</mergeCells>'
        '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
        '<pageSetup orientation="landscape" paperSize="8"/></worksheet>')


STYLES = '''<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.0000;-0.0000;0.0000"/></numFmts>
<fonts count="3"><font><sz val="11"/><name val="Arial"/><color rgb="FF243247"/></font><font><b/><sz val="15"/><name val="Arial"/><color rgb="FFFFFFFF"/></font><font><b/><sz val="11"/><name val="Arial"/><color rgb="FFFFFFFF"/></font></fonts>
<fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF253B53"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFF0F4F8"/><bgColor indexed="64"/></patternFill></fill></fills>
<borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="8">
<xf fontId="0" fillId="0" borderId="0" numFmtId="0" xfId="0"/>
<xf fontId="1" fillId="2" borderId="0" numFmtId="0" xfId="0" applyAlignment="1"><alignment vertical="center"/></xf>
<xf fontId="2" fillId="2" borderId="0" numFmtId="0" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
<xf fontId="0" fillId="0" borderId="0" numFmtId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
<xf fontId="0" fillId="0" borderId="0" numFmtId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
<xf fontId="0" fillId="3" borderId="0" numFmtId="0" xfId="0" applyAlignment="1"><alignment vertical="center" wrapText="1"/></xf>
<xf fontId="0" fillId="0" borderId="0" numFmtId="164" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
<xf fontId="0" fillId="3" borderId="0" numFmtId="164" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''


def build_excel(rows, root):
    latest = latest_metrics(rows)
    index = {(r['Kelas'], r['Model ID'], r['Metode']): r for r in latest}
    pairs = sorted({(r['Kelas'], r['Model ID']) for r in latest})
    sheets = []
    labels = ['PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓', 'Chamfer (cm) ↓',
              'Precision @2cm (%) ↑', 'Recall @2cm (%) ↑', 'F1 @2cm (%) ↑',
              'Precision @5cm (%) ↑', 'Recall @5cm (%) ↑', 'F1 @5cm (%) ↑',
              'Peak VRAM (GB)', 'Training tahap (menit)']
    block = ['Waktu hasil terakhir', 'Gambar registered', 'Gambar train', 'Gambar test', *labels]
    for experiment, title in [('baseline', 'Baseline'), ('scaffold_anchor', 'Studi ablasi')]:
        suffix = '' if experiment == 'baseline' else '_scaffold_anchor'
        values = []
        for cls, mdl in pairs:
            for method in ('3DGS', 'SuGaR'):
                row = index.get((cls, mdl, method + suffix))
                if row is None:
                    continue
                values.append([cls, mdl, row.get('Nama Model', mdl), method,
                               row.get('Timestamp'), *image_counts(root, cls, mdl, experiment, row),
                               *[number(row.get(k)) for k in METRICS]])
        note = 'Hasil terakhir per patung. ↑ lebih tinggi lebih baik; ↓ lebih rendah lebih baik. — = belum tersedia. Waktu/VRAM per tahap; SuGaR tidak termasuk 3DGS; ablasi tidak termasuk Scaffold.'
        headers = ['Kelas', 'Model ID', 'Nama patung', 'Metode', *block]
        sheets.append((title, sheet_xml(title + ' | Evaluasi rekonstruksi', note, headers, values,
            [(1, 5, 'Patung dan metode'), (6, 8, 'Jumlah gambar'), (9, 11, 'Kualitas render'),
             (12, 18, 'Kualitas geometri'), (19, 20, 'Sumber daya')])) )
    delta_rows = []
    for cls, mdl in pairs:
        for method in ('3DGS', 'SuGaR'):
            a, b = index.get((cls, mdl, method)), index.get((cls, mdl, method + '_scaffold_anchor'))
            if not a or not b:
                continue
            delta_rows.append([cls, mdl, a.get('Nama Model', mdl), method, *[
                number(b.get(k)) - number(a.get(k)) if number(a.get(k)) is not None and number(b.get(k)) is not None else None
                for k in METRICS]])
    sheets.insert(0, ('Perbandingan', sheet_xml('Ablasi − baseline | Patung yang sama',
        'Selisih per patung/metode; positif menguntungkan untuk ↑, negatif untuk ↓. VRAM/waktu: negatif lebih hemat. Bukan bukti signifikansi; pastikan split, resolusi dan konfigurasi sama.',
        ['Kelas', 'Model ID', 'Nama patung', 'Metode', *['Δ ' + x for x in labels]], delta_rows,
        [(1, 4, 'Patung dan metode'), (5, 7, 'Selisih kualitas render'),
         (8, 14, 'Selisih kualitas geometri'), (15, 16, 'Selisih sumber daya')])) )
    notes = [
        ['Cakupan', 'Semua baseline dan ablasi, tidak mengikuti filter tab/metode dashboard. Hanya status COMPLETED; hasil terakhir digabung per patung/metode.'],
        ['Sumber metrik', 'Log evaluasi summary_comparison.csv yang juga digunakan dashboard. Nilai kosong tidak diubah menjadi nol.'],
        ['Registered / train / test', 'Jumlah kamera pada run evaluasi, dari evaluation/metrics.json yang cocok dengan metrik. Registered = train + test. Train adalah gambar latih; test adalah gambar uji.'],
        ['Run lama', 'Jika jumlah registered/train belum tercatat, kolom tetap kosong (—). Jumlah test bisa diperoleh dari per_view. Evaluasi visual ulang mencatat ketiganya.'],
        ['Waktu dan VRAM', 'Diambil dari baris training, bukan evaluasi. Waktu SuGaR hanya tahap SuGaR. Biaya Scaffold belum masuk. Jangan menjumlahkan peak VRAM antar tahap.'],
        ['Perbandingan adil', 'Bandingkan patung, split uji, resolusi dan konfigurasi yang sama. Selisih dihitung dari nilai log yang telah dibulatkan.'],
        ['Kesimpulan', 'PSNR/SSIM/Precision/Recall/F1 lebih tinggi lebih baik; LPIPS/Chamfer lebih rendah lebih baik. Nilai render tidak menggantikan evaluasi geometri.'],
        ['Pengulangan', 'Lakukan beberapa seed sebelum menyimpulkan peningkatan konsisten. Rata-rata lintas eksperimen harus memakai kumpulan patung yang sama.'],
    ]
    sheets.append(('Panduan', sheet_xml('Cara membaca laporan', 'Laporan perbandingan hasil rekonstruksi 3D',
                                       ['Topik', 'Keterangan', ''], [[a, b, ''] for a, b in notes])))
    # Notes need a wide prose column; keep the report tables independently sized.
    sheets[-1] = (sheets[-1][0], sheets[-1][1].replace('min="2" max="2" width="19"', 'min="2" max="2" width="120"').replace('ht="26"', 'ht="46"').replace('vertical="center"/>', 'vertical="center" wrapText="1"/>'))
    ns = 'http://schemas.openxmlformats.org/'
    stream = io.BytesIO()
    with ZipFile(stream, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', f'<Types xmlns="{ns}package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets)+1)) + '</Types>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{ns}package/2006/relationships"><Relationship Id="rId1" Type="{ns}officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}spreadsheetml/2006/main" xmlns:r="{ns}officeDocument/2006/relationships"><sheets>' + ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{ns}package/2006/relationships">' + ''.join(f'<Relationship Id="rId{i}" Type="{ns}officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets)+1)) + f'<Relationship Id="styles" Type="{ns}officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', STYLES)
        for i, (_, xml) in enumerate(sheets, 1):
            archive.writestr(f'xl/worksheets/sheet{i}.xml', xml)
    return stream.getvalue()
