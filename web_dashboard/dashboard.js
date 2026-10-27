let activeFilter = 'ALL';
let cachedStatus = {};
let totalModels = 18;
let lastJobId = '';
let selectedLogId = null;

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtN = value => Number.isFinite(parseFloat(value)) ? parseFloat(value).toFixed(4) : '-';
const fmtSec = seconds => [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60].map(n => String(n).padStart(2, '0')).join(':');

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

window.addEventListener('DOMContentLoaded', async () => {
  try {
    const catalog = await api('/api/catalog');
    totalModels = catalog.total_models;
    $('kpiTotal').textContent = totalModels;
    $('btnRunAll').textContent = `Antrekan Semua ${totalModels} Model`;
    const filters = $('filterBar');
    filters.innerHTML = '';
    for (const cls of ['ALL', ...Object.keys(catalog.classes)]) {
      const button = document.createElement('button');
      button.className = 'filter-pill' + (cls === 'ALL' ? ' active' : '');
      button.textContent = cls === 'ALL' ? 'Semua' : cls;
      button.dataset.filter = cls;
      button.onclick = () => filterClass(cls);
      filters.appendChild(button);
    }
    await refreshAll();
  } catch (error) { showToast(error.message); }
  setInterval(pollLogs, 2000);
  setInterval(() => { refreshStatus(false); refreshMetrics(); }, 10000);
});

async function refreshAll() {
  await Promise.all([refreshStatus(false), refreshMetrics(), pollLogs()]);
}

async function refreshStatus(feedback) {
  try {
    cachedStatus = await api('/api/status');
    renderGrid(cachedStatus);
    const entries = Object.values(cachedStatus).flatMap(models => Object.values(models));
    const done = entries.filter(info => info.overall_status === 'Completed').length;
    $('kpiCompleted').textContent = `${done} / ${totalModels}`;
    $('kpiCompletedSub').textContent = `${totalModels - done} belum selesai training + evaluasi`;
    if (feedback) showToast('Status diperbarui');
  } catch (error) { console.error(error); }
}

function chip(done, yes = 'Selesai', no = 'Menunggu') {
  return `<span class="chip ${done ? 'chip-done' : 'chip-pending'}">${done ? yes : no}</span>`;
}

function renderGrid(data) {
  $('modelGrid').innerHTML = '';
  for (const [cls, models] of Object.entries(data)) {
    if (activeFilter !== 'ALL' && activeFilter !== cls) continue;
    const card = document.createElement('div');
    card.className = 'class-card';
    let cells = '';
    for (const [mdl, info] of Object.entries(models)) {
      const target = `'${cls}','${mdl}'`;
      cells += `<div class="model-cell">
        <div class="model-label">${esc(mdl)} ${info.has_video ? chip(true, 'Video') : ''}</div>
        <div class="pipeline-rows">
          <div class="pipeline-row"><span>Foto input</span>${chip(info.images_count > 0, `${info.images_count} foto`, 'Belum ada foto')}</div>
          <div class="pipeline-row"><span>COLMAP</span>${chip(info.colmap_done)}</div>
          <div class="pipeline-row"><span>3DGS</span>${chip(info['3dgs_30k'], '30k iter', info['3dgs_7k'] ? '7k iter' : 'Menunggu')}</div>
          <div class="pipeline-row"><span>SuGaR + OBJ</span>${chip(info.sugar_obj, 'OBJ lengkap', info.sugar_mesh ? 'Mesh kasar' : 'Menunggu')}</div>
          <div class="pipeline-row"><span>Evaluasi visual</span>${chip(info.evaluated)}</div>
        </div>
        <div class="dataset-path">Input: <code>${esc(info.input_path)}</code><br>Output: <code>${esc(info.output_path)}</code></div>
        <div class="step-buttons">
          <button class="btn btn-primary btn-sm btn-full" onclick="triggerRun('full_pipeline',${target})">Antrekan Pipeline Lengkap</button>
          <button class="btn-step" onclick="triggerRun('colmap',${target})">2. COLMAP</button>
          <button class="btn-step" onclick="triggerRun('3dgs',${target})">3. Latih 3DGS</button>
          <button class="btn-step" onclick="triggerRun('sugar',${target})">4. Latih SuGaR</button>
          <button class="btn-step" onclick="triggerRun('evaluate_visual',${target})">5. Evaluasi Visual</button>
          ${info.has_video ? `<button class="btn-step" onclick="triggerRun('extract_frames',${target})">1. Ekstrak Frame</button>` : ''}
          ${info.has_gt_lidar ? `<button class="btn-step" onclick="triggerRun('evaluate',${target})">Evaluasi LiDAR</button>` : ''}
        </div><div class="preview-box" id="preview-box-${cls}-${mdl}"></div></div>`;
    }
    card.innerHTML = `<div class="class-header"><span class="class-name">${esc(cls)}</span>
      <button class="btn btn-ghost btn-sm" onclick="triggerRun('full_pipeline','${cls}','')">Antrekan Kelas</button></div>
      <div class="class-body">${cells}</div>`;
    $('modelGrid').appendChild(card);
    for (const mdl of Object.keys(models)) loadPreview(cls, mdl);
  }
}

function filterClass(cls) {
  activeFilter = cls;
  document.querySelectorAll('.filter-pill').forEach(button => button.classList.toggle('active', button.dataset.filter === cls));
  renderGrid(cachedStatus);
}

async function triggerRun(action, cls, mdl) {
  const payload = {action, class_name: cls, model_id: mdl,
    gpu: Number($('gpuId').value), refinement_time: $('refinementTime').value,
    vertices: Number($('meshTriangles').value), resume: $('resumeCompleted').checked,
    evaluation_method: $('evaluationMethod').value};
  if (action === 'evaluate') {
    const text = $('scaleFactor').value.trim();
    payload.scale_factor = Number(text);
    if (!text || !Number.isFinite(payload.scale_factor) || payload.scale_factor <= 0) {
      showToast('Isi faktor kalibrasi meter per unit model untuk evaluasi LiDAR.');
      return;
    }
  }
  try {
    const results = [];
    if (action === 'evaluate_visual') {
      for (const method of ['3DGS', 'SuGaR']) results.push(await api('/api/run', {...payload, evaluation_method: method}));
    } else results.push(await api('/api/run', payload));
    const added = results.reduce((n, result) => n + result.added.length, 0);
    const skipped = results.reduce((n, result) => n + result.skipped_empty.length, 0);
    showToast(`${added} pekerjaan masuk antrean${skipped ? `; ${skipped} dataset kosong dilewati` : ''}.`);
    await pollLogs();
  } catch (error) { showToast(error.message); }
}

async function queueControl(action, data = {}) {
  try {
    const response = await api('/api/queue/' + action, data);
    showToast(response.message || `${response.cancelled} pekerjaan dibatalkan.`);
    await pollLogs();
  } catch (error) { showToast(error.message); }
}

async function triggerStop() {
  try { showToast((await api('/api/stop', {})).message); await pollLogs(); }
  catch (error) { showToast(error.message); }
}

async function pollLogs() {
  try {
    const data = await api('/api/queue');
    $('btnStop').disabled = !data.is_running;
    $('jobStrip').classList.toggle('hidden', !data.is_running);
    $('jobLabel').textContent = data.action || '';
    $('jobTarget').textContent = data.target || '';
    $('jobTimer').textContent = fmtSec(data.elapsed_seconds);
    $('termStatus').textContent = data.status.toLowerCase();
    $('termStatus').className = 'chip ' + (data.status === 'COMPLETED' ? 'chip-done' : data.status === 'FAILED' ? 'chip-warn' : 'chip-partial');
    $('queueSummary').textContent = `${data.paused ? 'Dijeda' : 'Aktif'} · ${data.counts.RUNNING || 0} berjalan · ${data.counts.QUEUED || 0} menunggu · ${data.counts.COMPLETED || 0} selesai · ${data.counts.FAILED || 0} gagal`;
    $('btnPause').textContent = data.paused ? 'Lanjutkan Antrean' : 'Jeda Antrean';
    $('btnPause').onclick = () => queueControl(data.paused ? 'resume' : 'pause');
    $('queueBody').innerHTML = '';
    const jobs = [...data.jobs.filter(j => ['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status)),
                  ...data.jobs.filter(j => !['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status)).reverse().slice(0, 30)];
    for (const job of jobs) {
      const row = document.createElement('tr');
      const failed = ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(job.status);
      row.innerHTML = `<td>${esc(job.class_name)}/${esc(job.model_id)}</td><td>${esc(job.action)}${job.action.includes('evaluate') ? ' · ' + esc(job.evaluation_method) : ''}</td>
        <td>${chip(job.status === 'COMPLETED', esc(job.status), esc(job.status))}</td>
        <td>${job.started_at ? fmtSec((job.finished_at || Date.now() / 1000) - job.started_at) : '-'}</td><td></td>`;
      const logButton = document.createElement('button');
      logButton.className = 'btn btn-ghost btn-sm';
      logButton.textContent = 'Log';
      logButton.onclick = () => showJobLogs(job.id);
      row.lastElementChild.appendChild(logButton);
      if (job.status === 'QUEUED' || failed) {
        const button = document.createElement('button');
        button.className = 'btn btn-ghost btn-sm';
        button.textContent = failed ? 'Antrekan Ulang' : 'Batalkan';
        button.onclick = failed ? async () => {
          try { await api('/api/run', job); await pollLogs(); } catch (error) { showToast(error.message); }
        } : () => queueControl('cancel', {job_id: job.id});
        row.lastElementChild.appendChild(button);
      }
      if (job.error) row.title = job.error;
      $('queueBody').appendChild(row);
    }
    if (!jobs.length) $('queueBody').innerHTML = '<tr><td colspan="5" class="empty-msg">Belum ada pekerjaan. Tambahkan model ke antrean.</td></tr>';
    if (!selectedLogId && data.logs.length) renderLogs(data.logs);
    const signature = data.jobs.map(j => j.id + j.status).join();
    if (signature !== lastJobId) { lastJobId = signature; refreshStatus(false); refreshMetrics(); }
  } catch (error) { console.error(error); }
}

async function showJobLogs(id) {
  try {
    const data = await api(`/api/jobs/${id}/logs`);
    selectedLogId = id;
    $('termLabel').textContent = `${data.job.class_name}/${data.job.model_id} · ${data.job.status}`;
    renderLogs(data.logs.length ? data.logs : ['Belum ada log untuk pekerjaan ini.']);
  } catch (error) { showToast(error.message); }
}
function showLiveLogs() { selectedLogId = null; $('termLabel').textContent = 'output'; pollLogs(); }

function renderLogs(logs) {
  $('termBody').innerHTML = '';
  for (const line of logs) {
    const div = document.createElement('div');
    div.className = 'log-line' + (/ERROR|GAGAL|Exception/.test(line) ? ' l-err' : /QUEUE|RUN/.test(line) ? ' l-sys' : '');
    div.textContent = line;
    $('termBody').appendChild(div);
  }
  if ($('chkScroll').checked) $('termBody').scrollTop = $('termBody').scrollHeight;
}
function clearTerm() { $('termBody').textContent = ''; }
async function copyLogs() {
  try { await navigator.clipboard.writeText($('termBody').innerText); showToast('Log disalin.'); }
  catch (error) { showToast(error.message); }
}

async function refreshMetrics() {
  try {
    const rows = (await api('/api/metrics')).filter(row => row['Metode'] === '3DGS' || row['Metode'] === 'SuGaR');
    const latest = new Map();
    for (const row of rows) {
      if (row.Status !== 'COMPLETED') continue;
      const key = [row.Kelas, row['Model ID'], row.Metode].join('/');
      const current = latest.get(key) || {};
      // Training rows, visual evaluation and geometry evaluation stay linked by model/method.
      const evaluated = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)'].some(column => Number.isFinite(parseFloat(row[column])));
      for (const [column, value] of Object.entries(row)) {
        if (evaluated && ['Total Waktu (Menit)', 'Peak VRAM (GB)'].includes(column)) continue;
        if (value !== '-' && value !== '') current[column] = value;
      }
      latest.set(key, current);
    }
    const display = [...latest.values()];
    const columns = ['Timestamp', 'Kelas', 'Model ID', 'Metode', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)', 'Peak VRAM (GB)', 'Total Waktu (Menit)'];
    $('metricsBody').innerHTML = display.map(row => '<tr>' + columns.map((column, i) => `<td>${esc(i < 4 ? row[column] || '-' : fmtN(row[column]))}</td>`).join('') + '</tr>').join('') || '<tr><td colspan="10" class="empty-msg">Belum ada hasil. Pipeline lengkap mencakup evaluasi visual.</td></tr>';
    for (const [column, target, unit] of [['PSNR (dB)', 'kpiPsnr', ' dB'], ['Chamfer Dist (cm)', 'kpiChamfer', ' cm']]) {
      const values = display.map(row => Number.parseFloat(row[column])).filter(Number.isFinite);
      $(target).textContent = values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2) + unit : '—';
    }
  } catch (error) { console.error(error); }
}
function exportCSV() { window.open('/api/metrics.csv', '_blank'); }
function showToast(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  setTimeout(() => $('toast').classList.remove('show'), 6500);
}

async function loadPreview(cls, mdl) {
  try {
    const data = await api(`/api/previews/${cls}/${mdl}`);
    const box = $(`preview-box-${cls}-${mdl}`);
    if (!box) return;
    box.innerHTML = '';
    for (const item of data.images) {
      const img = document.createElement('img');
      img.className = 'preview-thumb'; img.src = item.src; img.alt = item.label; img.loading = 'lazy';
      img.onclick = () => openLightbox(item.src, item.label);
      box.appendChild(img);
    }
  } catch (error) { console.error(error); }
}
function openLightbox(src, caption) { $('lightboxImg').src = src; $('lightboxCaption').textContent = caption; $('lightbox').classList.add('open'); }
function closeLightbox() { $('lightbox').classList.remove('open'); }
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeLightbox(); });
