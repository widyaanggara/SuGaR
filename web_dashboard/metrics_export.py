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
METHODS = ('2DGS_scaffold_anchor', 'RaDe-GS_scaffold_anchor', '2DGS', 'RaDe-GS', '3DGS', 'SuGaR', '3DGS_scaffold_anchor', 'SuGaR_scaffold_anchor')


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
    method = row.get('Metode', '').replace('_scaffold_anchor', '').replace('-', '').lower()
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
    # Legacy evaluation uses this experiment's 3DGS cameras.json for BOTH methods.
    # Match the actual test image names and split before recovering its population.
    views = report.get('per_view')
    recovered = legacy_image_counts(base / '3dgs' / cls / model / 'cameras.json', report)
    if recovered is not None:
        return recovered
    return [None, None, len(views) if isinstance(views, list) else None]


def legacy_image_counts(camera_path, report):
    cameras = read_json(camera_path)
    views = report.get('per_view')
    interval = report.get('eval_split_interval')
    if (not isinstance(cameras, list) or not cameras or not isinstance(views, list) or not views
            or type(interval) is not int or interval < 2):
        return None
    if any(not isinstance(c, dict) or not isinstance(c.get('img_name'), str) or not c['img_name']
           for c in cameras):
        return None
    if any(not isinstance(v, dict) or not isinstance(v.get('image'), str) or not v['image']
           for v in views):
        return None
    # load_gs_cameras deduplicates by img_name, then sorts before splitting.
    names = sorted({c['img_name'] for c in cameras})
    tested = [v['image'] for v in views]
    if len(set(tested)) != len(tested) or sorted(tested) != names[::interval]:
        return None
    return [len(names), len(names) - len(tested), len(tested)]


def column_name(index):
    result = ''
    while index:
        index, rem = divmod(index - 1, 26)
        result = chr(65 + rem) + result
    return result


def xml_text(value):
    return escape(re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value)))


# ---------------------------------------------------------------------------
# Workbook layout
# ---------------------------------------------------------------------------
METHOD_ORDER = ('3DGS', 'SuGaR', '2DGS', 'RaDe-GS')
# (source column, header, number format, direction: 1 higher is better, -1 lower is better)
METRIC_COLUMNS = [
    ('PSNR (dB)', 'PSNR (dB) ↑', '0.00', 1),
    ('SSIM', 'SSIM ↑', '0.0000', 1),
    ('LPIPS', 'LPIPS ↓', '0.0000', -1),
    ('Chamfer Dist (cm)', 'Chamfer (cm) ↓', '0.00', -1),
    ('Precision @2cm (%)', 'Precision (%) ↑', '0.0', 1),
    ('Recall @2cm (%)', 'Recall (%) ↑', '0.0', 1),
    ('F1 @2cm (%)', 'F1 (%) ↑', '0.0', 1),
    ('Precision @5cm (%)', 'Precision (%) ↑', '0.0', 1),
    ('Recall @5cm (%)', 'Recall (%) ↑', '0.0', 1),
    ('F1 @5cm (%)', 'F1 (%) ↑', '0.0', 1),
    ('Peak VRAM (GB)', 'Peak VRAM (GB) ↓', '0.00', -1),
    ('Total Waktu (Menit)', 'Waktu training (menit) ↓', '0.0', -1),
]
INK, MUTED = 'FF292524', 'FF78716C'


class Styles:
    """Small registry so each sheet can ask for exactly the cell look it needs."""

    def __init__(self):
        self.fonts = ['<font><sz val="10"/><name val="Arial"/><color rgb="FF292524"/></font>']
        self.fills = ['<fill><patternFill patternType="none"/></fill>', '<fill><patternFill patternType="gray125"/></fill>']
        self.borders = ['<border><left/><right/><top/><bottom/><diagonal/></border>']
        self.formats = {}
        self.xfs = ['<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>']
        self.cache = {}

    @staticmethod
    def _index(items, value):
        if value not in items:
            items.append(value)
        return items.index(value)

    def get(self, bold=False, size=10, color=INK, fill=None, fmt=None, align='left', wrap=False, top=None, bottom=None):
        key = (bold, size, color, fill, fmt, align, wrap, top, bottom)
        if key in self.cache:
            return self.cache[key]
        font = self._index(self.fonts, f'<font>{"<b/>" if bold else ""}<sz val="{size}"/><name val="Arial"/><color rgb="{color}"/></font>')
        fill_id = 0 if fill is None else self._index(self.fills, f'<fill><patternFill patternType="solid"><fgColor rgb="{fill}"/><bgColor indexed="64"/></patternFill></fill>')
        side = lambda name, colour: f'<{name} style="thin"><color rgb="{colour}"/></{name}>' if colour else f'<{name}/>'
        border = self._index(self.borders, f'<border><left/><right/>{side("top", top)}{side("bottom", bottom)}<diagonal/></border>')
        fmt_id = 0
        if fmt == '0':
            fmt_id = 1
        elif fmt:
            fmt_id = self.formats.setdefault(fmt, 164 + len(self.formats))
        wrap_attr = ' wrapText="1"' if wrap else ''
        xf = (f'<xf numFmtId="{fmt_id}" fontId="{font}" fillId="{fill_id}" borderId="{border}" xfId="0" '
              f'applyNumberFormat="1" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1">'
              f'<alignment horizontal="{align}" vertical="center"{wrap_attr}/></xf>')
        self.cache[key] = self._index(self.xfs, xf)
        return self.cache[key]

    def xml(self):
        formats = ''.join(f'<numFmt numFmtId="{i}" formatCode="{xml_text(code)}"/>' for code, i in self.formats.items())
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f'<numFmts count="{len(self.formats)}">{formats}</numFmts>'
                f'<fonts count="{len(self.fonts)}">{"".join(self.fonts)}</fonts>'
                f'<fills count="{len(self.fills)}">{"".join(self.fills)}</fills>'
                f'<borders count="{len(self.borders)}">{"".join(self.borders)}</borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                f'<cellXfs count="{len(self.xfs)}">{"".join(self.xfs)}</cellXfs>'
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')


def sheet_xml(styles, title, note, columns, rows, groups=(), freeze=4, empty='Belum ada hasil.', autofilter=True):
    """columns: [(header, width, number format or None, align)];
    rows: [{'values': [...], 'band': 0|1, 'start': bool}]."""
    width = len(columns)
    last = column_name(width)
    xml_rows = []

    def cell(ref, value, style):
        if value is None or value == '':
            value = '—'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'<c r="{ref}" s="{style}"><v>{value}</v></c>'
        return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{xml_text(value)}</t></is></c>'

    title_style = styles.get(bold=True, size=14)
    note_style = styles.get(size=9, color=MUTED, wrap=True)
    xml_rows.append(f'<row r="1" ht="26" customHeight="1">{cell("A1", title, title_style)}</row>')
    xml_rows.append(f'<row r="2" ht="30" customHeight="1">{cell("A2", note, note_style)}</row>')
    merges = [f'A1:{last}1', f'A2:{last}2']

    group_style = styles.get(bold=True, size=9, color=INK, fill='FFE7E5E4', align='center', bottom='FFD6D3D1')
    group_cells = [f'<c r="{column_name(c)}3" s="{group_style}"/>' for c in range(1, width + 1)]
    for start, stop, label in groups:
        group_cells[start - 1] = cell(f'{column_name(start)}3', label, group_style)
        if stop > start:
            merges.append(f'{column_name(start)}3:{column_name(stop)}3')
    xml_rows.append(f'<row r="3" ht="20" customHeight="1">{"".join(group_cells)}</row>')

    header_cells = []
    for c, (header, _, fmt, align) in enumerate(columns, 1):
        style = styles.get(bold=True, size=9, color='FFFFFFFF', fill='FF1C1917', align=align, wrap=True)
        header_cells.append(cell(f'{column_name(c)}4', header, style))
    xml_rows.append(f'<row r="4" ht="40" customHeight="1">{"".join(header_cells)}</row>')

    if not rows:
        message_style = styles.get(color=MUTED)
        xml_rows.append(f'<row r="5" ht="22" customHeight="1">{cell("A5", empty, message_style)}</row>')
        merges.append(f'A5:{last}5')
    for r, row in enumerate(rows, 5):
        fill = 'FFF5F5F4' if row.get('band') else None
        top = 'FFD6D3D1' if row.get('start') and r > 5 else None
        contents = []
        for c, value in enumerate(row['values'], 1):
            _, _, fmt, align = columns[c - 1]
            color = MUTED if value is None else INK
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            style = styles.get(color=color, fill=fill,
                               fmt=fmt if numeric else None, align=align if value is not None else 'center' if align == 'right' else align,
                               top=top, wrap=row.get('wrap', False))
            contents.append(cell(f'{column_name(c)}{r}', value, style))
        height = row.get('height', 20)
        xml_rows.append(f'<row r="{r}" ht="{height}" customHeight="1">{"".join(contents)}</row>')

    data_end = max(5, 4 + len(rows))
    cols = ''.join(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>' for i, (_, w, _, _) in enumerate(columns, 1))
    pane = (f'<pane xSplit="{freeze}" ySplit="4" topLeftCell="{column_name(freeze + 1)}5" activePane="bottomRight" state="frozen"/>'
            if freeze else '<pane ySplit="4" topLeftCell="A5" activePane="bottomLeft" state="frozen"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:{last}{data_end}"/>'
            f'<sheetViews><sheetView workbookViewId="0" showGridLines="0">{pane}</sheetView></sheetViews>'
            f'<cols>{cols}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
            + (f'<autoFilter ref="A4:{last}{data_end}"/>' if autofilter and rows else '')
            + f'<mergeCells count="{len(merges)}">' + ''.join(f'<mergeCell ref="{ref}"/>' for ref in merges) + '</mergeCells>'
            '<pageMargins left="0.3" right="0.3" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>'
            '<pageSetup orientation="landscape" paperSize="9"/></worksheet>')


def build_excel(rows, root, patung=None):
    """patung: optional [(kelas, model_id, nama)] so every registered statue is listed, even without results."""
    latest = latest_metrics(rows)
    index = {(r['Kelas'], r['Model ID'], r['Metode']): r for r in latest}
    if patung is None:
        patung = [(cls, mdl, next((r.get('Nama Model') for r in latest
                                   if (r['Kelas'], r['Model ID']) == (cls, mdl) and r.get('Nama Model')), mdl))
                  for cls, mdl in sorted({(r['Kelas'], r['Model ID']) for r in latest})]
    styles = Styles()
    sheets = []
    experiments = [('baseline', 'Baseline', ''), ('scaffold_anchor', 'Studi ablasi', '_scaffold_anchor')]

    # One tab per method: every statue, every column recorded for that method.
    quality = METRIC_COLUMNS[:10]
    columns = [('Eksperimen', 12, None, 'left'), ('Kelas', 16, None, 'left'), ('Model ID', 19, None, 'left'),
               ('Nama patung', 20, None, 'left'),
               *[(header, 12, fmt, 'right') for _, header, fmt, _ in quality],
               ('Jumlah titik/vertex', 13, '#,##0', 'right'), ('Peak VRAM (GB)', 11, '0.00', 'right'),
               ('Waktu training (menit)', 11, '0.0', 'right'),
               ('Gambar registered', 11, '0', 'right'), ('Gambar train', 10, '0', 'right'), ('Gambar test', 10, '0', 'right'),
               ('Status', 12, None, 'left'), ('Session', 28, None, 'left'), ('Waktu hasil terakhir', 19, None, 'left')]
    groups = [(1, 4, 'Patung'), (5, 7, 'Kualitas render'), (8, 11, 'Geometri @2 cm'), (12, 14, 'Geometri @5 cm'),
              (15, 17, 'Model & sumber daya'), (18, 20, 'Jumlah gambar'), (21, 23, 'Catatan run')]
    for method in METHOD_ORDER:
        table = []
        for band, (cls, mdl, name) in enumerate(patung):
            group = []
            for experiment, title, suffix in experiments:
                row = index.get((cls, mdl, method + suffix))
                if not row and (suffix or group):
                    continue  # ablation rows only appear once that experiment exists
                row = row or {}
                group.append({'values': [title, cls, mdl, row.get('Nama Model', name),
                                         *[number(row.get(k)) for k, *_ in quality],
                                         number(row.get('Jumlah Titik/Vertex')), number(row.get('Peak VRAM (GB)')),
                                         number(row.get('Total Waktu (Menit)')),
                                         *(image_counts(root, cls, mdl, experiment, row) if row else [None] * 3),
                                         row.get('Status'), row.get('Session'), row.get('Timestamp')],
                              'band': band % 2, 'start': not group})
            table += group
        sheets.append((method, sheet_xml(styles, f'{method} · Hasil rekonstruksi',
            'Hasil terakhir per patung (status COMPLETED). ↑ lebih tinggi lebih baik, ↓ lebih rendah lebih baik. '
            '— = belum tersedia. Riwayat lengkap setiap run ada di tab Semua log.',
            columns, table, groups, empty=f'Belum ada hasil {method}.')))

    # Every recorded row from the beginning, all columns, oldest first.
    preferred = ['Timestamp', 'Session', 'Kelas', 'Model ID', 'Nama Model', 'Metode', 'Status',
                 *[k for k, *_ in METRIC_COLUMNS[:10]], 'Jumlah Titik/Vertex', 'Peak VRAM (GB)', 'Total Waktu (Menit)']
    keys = preferred + sorted({k for r in rows for k in r} - set(preferred))
    keys = [k for k in keys if any(r.get(k) not in (None, '', '-') for r in rows)] or preferred
    formats = {k: fmt for k, _, fmt, _ in METRIC_COLUMNS}
    formats['Jumlah Titik/Vertex'] = '#,##0'
    log_columns = [(k, 19 if k == 'Timestamp' else 28 if k == 'Session' else 12,
                    formats.get(k), 'right' if k in formats else 'left') for k in keys]
    history = []
    for i, row in enumerate(sorted(rows, key=lambda r: r.get('Timestamp', ''))):
        values = []
        for k in keys:
            value = row.get(k)
            value = None if value in ('-', '') else value
            values.append(number(value) if k in formats and number(value) is not None else value)
        history.append({'values': values, 'band': i % 2})
    sheets.append(('Semua log', sheet_xml(styles, 'Semua log evaluasi dan training',
        'Seluruh baris yang tercatat dari awal, termasuk run lama, gagal, dan studi ablasi, urut dari yang paling lama.',
        log_columns, history, [(1, len(log_columns), 'Log')], freeze=0, empty='Belum ada log.')))

    ns = 'http://schemas.openxmlformats.org/'
    stream = io.BytesIO()
    with ZipFile(stream, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', f'<Types xmlns="{ns}package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, len(sheets)+1)) + '</Types>')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{ns}package/2006/relationships"><Relationship Id="rId1" Type="{ns}officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}spreadsheetml/2006/main" xmlns:r="{ns}officeDocument/2006/relationships"><sheets>' + ''.join(f'<sheet name="{xml_text(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, (name, _) in enumerate(sheets, 1)) + '</sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{ns}package/2006/relationships">' + ''.join(f'<Relationship Id="rId{i}" Type="{ns}officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(sheets)+1)) + f'<Relationship Id="styles" Type="{ns}officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', styles.xml())
        for i, (_, xml) in enumerate(sheets, 1):
            archive.writestr(f'xl/worksheets/sheet{i}.xml', xml)
    return stream.getvalue()
