let activeFilter = 'ALL';
let cachedStatus = {};
let totalModels = 18;
let lastJobId = '';
let selectedLogId = null;
let selectedLogRequest = 0;
let selectedLogLoading = false;
let lastQueueSignature = null;
let lastGridSignature = null;
let cachedMetrics = [];

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtN = value => Number.isFinite(parseFloat(value)) ? parseFloat(value).toFixed(4) : '-';
const fmtSec = seconds => [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60].map(n => String(n).padStart(2, '0')).join(':');
const actionLabel = action => ({full_pipeline: 'Pipeline lengkap', colmap: 'Rekonstruksi COLMAP',
  '3dgs': 'Training 3DGS', sugar: 'Training SuGaR', extract_frames: 'Ekstraksi frame',
  evaluate_visual: 'Evaluasi visual', evaluate: 'Evaluasi LiDAR'}[action] || action || '');
const settingsTabs = ['resolution', 'colmap', 'training', 'lidar'];

function switchSettingsTab(name, focus = false) {
  if (!settingsTabs.includes(name)) return;
  for (const tab of settingsTabs) {
    const active = tab === name;
    $('tab-' + tab).setAttribute('aria-selected', String(active));
    $('tab-' + tab).tabIndex = active ? 0 : -1;
    $('pane-' + tab).hidden = !active;
  }
  if (focus) $('tab-' + name).focus();
}

function initializeNavigation() {
  for (const [index, name] of settingsTabs.entries()) {
    $('tab-' + name).onkeydown = event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % settingsTabs.length;
      if (event.key === 'ArrowLeft') next = (index + settingsTabs.length - 1) % settingsTabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = settingsTabs.length - 1;
      if (next !== undefined) { event.preventDefault(); switchSettingsTab(settingsTabs[next], true); }
    };
  }
  switchSettingsTab('resolution');
  const links = document.querySelectorAll('.side-nav a');
  links.forEach(link => { link.onclick = () => links.forEach(item => {
    item.classList.toggle('active', item === link);
    if (item === link) item.setAttribute('aria-current', 'location');
    else item.removeAttribute('aria-current');
  }); });
}

async function api(path, data) {
  const response = await fetch(path, data === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

const settingControls = ['gpuId', 'refinementTime', 'meshTriangles', 'resumeCompleted',
  'colmapMaxImageSize', 'imageDownscale', 'imageMaxSize', 'colmapDevice', 'colmapMatcher',
  'colmapMaxFeatures', 'colmapSequentialOverlap', 'sourceWidth', 'sourceHeight',
  'gaussiansPerTriangle', 'regularization'];
let presets = {};
const sugarControlIds = {vertices: 'meshTriangles', gaussians_per_triangle: 'gaussiansPerTriangle',
  refinement_time: 'refinementTime', regularization: 'regularization'};

function applyPreset(name) {
  if (presets[name]) {
    for (const [key, value] of Object.entries(presets[name].values)) $(sugarControlIds[key]).value = String(value);
  }
  $('settingsPreset').value = presets[name] ? name : 'custom';
  updateSettings(false);
}

function updateSettings(custom = true) {
  if (custom) $('settingsPreset').value = 'custom';
  const preset = $('settingsPreset').value;
  $('presetDescription').textContent = presets[preset]?.description || 'Konfigurasi kustom/tersimpan lama. Parameter ini tidak diklaim sebagai preset resmi. Gunakan konfigurasi yang konsisten untuk eksperimen skripsi.';
  $('colmapSequentialOverlap').disabled = $('colmapMatcher').value !== 'sequential';
  const width = Number($('sourceWidth').value), height = Number($('sourceHeight').value);
  const downscale = Number($('imageDownscale').value), cap = Number($('imageMaxSize').value);
  const colmapCap = Number($('colmapMaxImageSize').value);
  if ([width, height, downscale, colmapCap].every(v => Number.isFinite(v) && v > 0) && Number.isFinite(cap) && cap >= 0) {
    const size = factor => `${Math.max(1, Math.round(width / factor))} × ${Math.max(1, Math.round(height / factor))}`;
    const trainingFactor = Math.max(downscale, cap ? Math.max(width, height) / cap : 1);
    $('resolutionPreview').textContent = `Acuan ${width} × ${height} px → fitur COLMAP ≈ ${size(Math.max(1, Math.max(width, height) / colmapCap))} px; training & evaluasi ≈ ${size(trainingFactor)} px. Foto asli tidak diubah.`;
  } else $('resolutionPreview').textContent = 'Isi ukuran acuan dan pengaturan resolusi yang valid untuk melihat perkiraan.';
  try {
    const values = Object.fromEntries(settingControls.map(id => [id, id === 'resumeCompleted' ? $(id).checked : $(id).value]));
    window.localStorage?.setItem('sugar-experiment-settings-v1', JSON.stringify({...values, settingsPreset: preset}));
  } catch (_) { /* The dashboard also works when browser storage is unavailable. */ }
}

function initializeSettings() {
  try {
    const saved = JSON.parse(window.localStorage?.getItem('sugar-experiment-settings-v1') || 'null');
    if (saved && typeof saved === 'object') {
      for (const id of [...settingControls, 'settingsPreset']) if (Object.hasOwn(saved, id)) {
        if (id === 'resumeCompleted') $(id).checked = saved[id] === true;
        else {
          const previous = $(id).value;
          $(id).value = String(saved[id]);
          if (!$(id).value) $(id).value = previous;
        }
      }
    }
  } catch (_) { /* Ignore corrupt/blocked browser storage. */ }
  if (!presets[$('settingsPreset').value]) $('settingsPreset').value = 'custom';
  else {
    const values = presets[$('settingsPreset').value].values;
    if (Object.entries(values).some(([key, value]) => String(value) !== $(sugarControlIds[key]).value)) {
      $('settingsPreset').value = 'custom';
    }
  }
  for (const id of settingControls) {
    const changed = () => updateSettings(Object.values(sugarControlIds).includes(id));
    $(id).oninput = changed;
    $(id).onchange = changed;
  }
  updateSettings(false);
}

async function checkColmap() {
  $('btnCheckColmap').disabled = true;
  $('colmapCapability').textContent = 'Memeriksa build COLMAP server...';
  try {
    const info = await api('/api/colmap');
    $('colmapCapability').textContent = `${info.available ? (info.cuda ? 'CUDA tersedia pada build COLMAP. ' : 'COLMAP CPU / CUDA belum terdeteksi. ') : 'COLMAP belum tersedia. '}${info.message || ''}`;
    $('colmapCapability').title = info.version || '';
  } catch (error) { $('colmapCapability').textContent = 'Pemeriksaan COLMAP gagal: ' + error.message; }
  finally { $('btnCheckColmap').disabled = false; }
}

window.addEventListener('DOMContentLoaded', async () => {
  initializeNavigation();
  try {
    const references = await api('/api/presets');
    presets = references.presets;
    const choices = $('settingsPreset');
    choices.innerHTML = '';
    for (const [key, preset] of Object.entries(presets)) {
      const option = document.createElement('option');
      option.value = key; option.textContent = preset.label;
      choices.appendChild(option);
    }
    const custom = document.createElement('option');
    custom.value = 'custom'; custom.textContent = 'Kustom / pengaturan tersimpan'; choices.appendChild(custom);
    choices.value = references.default;
    initializeSettings();
    checkColmap();
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
  lastGridSignature = null;
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
  const signature = JSON.stringify([activeFilter, data]);
  if (signature === lastGridSignature) return;
  lastGridSignature = signature;
  $('modelGrid').innerHTML = '';
  for (const [cls, models] of Object.entries(data)) {
    if (activeFilter !== 'ALL' && activeFilter !== cls) continue;
    const card = document.createElement('div');
    card.className = 'class-card';
    let cells = '';
    for (const [mdl, info] of Object.entries(models)) {
      const target = `'${cls}','${mdl}'`;
      const modelLabel = mdl.startsWith('model_') ? 'Model ' + mdl.slice(6) : 'Benchmark ' + mdl.split('_').at(-1);
      cells += `<div class="model-cell">
        <div class="model-label"><span>${esc(modelLabel)}</span>${info.has_video ? chip(true, 'Video') : `<span class="model-code">${esc(mdl)}</span>`}</div>
        <div class="pipeline-rows">
          <div class="pipeline-row"><span><span class="stage-index">01</span>Foto input</span>${chip(info.images_count > 0, `${info.images_count} foto`, 'Belum ada foto')}</div>
          <div class="pipeline-row"><span><span class="stage-index">02</span>COLMAP</span>${chip(info.colmap_done)}</div>
          <div class="pipeline-row"><span><span class="stage-index">03</span>3DGS</span>${chip(info['3dgs_30k'], '30k iter', info['3dgs_7k'] ? '7k iter' : 'Menunggu')}</div>
          <div class="pipeline-row"><span><span class="stage-index">04</span>SuGaR + OBJ</span>${chip(info.sugar_obj, 'OBJ lengkap', info.sugar_mesh ? 'Mesh kasar' : 'Menunggu')}</div>
          <div class="pipeline-row"><span><span class="stage-index">05</span>Evaluasi visual</span>${chip(info.evaluated)}</div>
        </div>
        <details class="dataset-path"><summary>Lihat folder input &amp; output</summary><code>${esc(info.input_path)}</code><code>${esc(info.output_path)}</code></details>
        <div class="step-buttons">
          <button class="btn btn-primary btn-sm btn-full" onclick="triggerRun('full_pipeline',${target})">+ Antrekan Pipeline Lengkap</button>
          <button class="btn-step" onclick="triggerRun('colmap',${target})">2. COLMAP</button>
          <button class="btn-step" onclick="triggerRun('3dgs',${target})">3. Latih 3DGS</button>
          <button class="btn-step" onclick="triggerRun('sugar',${target})">4. Latih SuGaR</button>
          <button class="btn-step" onclick="triggerRun('evaluate_visual',${target})">5. Evaluasi Visual</button>
          ${info.has_video ? `<button class="btn-step" onclick="triggerRun('extract_frames',${target})">1. Ekstrak Frame</button>` : ''}
          ${info.has_gt_lidar ? `<button class="btn-step" onclick="triggerRun('evaluate',${target})">Evaluasi LiDAR</button>` : ''}
        </div><div class="preview-box" id="preview-box-${cls}-${mdl}"></div></div>`;
    }
    const number = String(Object.keys(data).indexOf(cls) + 1).padStart(2, '0');
    card.innerHTML = `<div class="class-header"><div class="class-heading"><span class="class-number">${number}</span><div><div class="class-name">${esc(cls === 'TanksAndTemples' ? 'Tanks & Temples' : cls)}</div><div class="class-meta">${Object.keys(models).length} model · ${cls === 'TanksAndTemples' ? 'Benchmark rekonstruksi' : 'Tipologi Togog Bali'}</div></div></div>
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
    gaussians_per_triangle: Number($('gaussiansPerTriangle').value),
    regularization: $('regularization').value,
    colmap_max_image_size: Number($('colmapMaxImageSize').value),
    image_downscale: Number($('imageDownscale').value),
    image_max_size: Number($('imageMaxSize').value),
    colmap_device: $('colmapDevice').value,
    colmap_matcher: $('colmapMatcher').value,
    colmap_max_features: Number($('colmapMaxFeatures').value),
    colmap_sequential_overlap: Number($('colmapSequentialOverlap').value),
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
    $('jobLabel').textContent = actionLabel(data.action);
    $('jobTarget').textContent = data.target || '';
    $('jobTimer').textContent = fmtSec(data.elapsed_seconds);
    $('termStatus').textContent = data.status.toLowerCase();
    $('termStatus').className = 'chip ' + (data.status === 'COMPLETED' ? 'chip-done' : data.status === 'FAILED' ? 'chip-warn' : 'chip-partial');
    $('queueSummary').textContent = `${data.paused ? 'Dijeda' : 'Aktif'} · ${data.counts.RUNNING || 0} berjalan · ${data.counts.QUEUED || 0} menunggu · ${data.counts.COMPLETED || 0} selesai · ${data.counts.FAILED || 0} gagal`;
    $('btnPause').textContent = data.paused ? 'Lanjutkan Antrean' : 'Jeda Antrean';
    $('btnPause').onclick = () => queueControl(data.paused ? 'resume' : 'pause');
    const jobs = [...data.jobs.filter(j => ['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status)),
                  ...data.jobs.filter(j => !['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status)).reverse().slice(0, 30)];
    const queueSignature = JSON.stringify(jobs.map(job => [job.id, job.status, job.error]));
    // Keep buttons in the DOM while the timer ticks, so polling cannot swallow a click.
    if (queueSignature !== lastQueueSignature) {
      $('queueBody').innerHTML = '';
      for (const job of jobs) {
        const row = document.createElement('tr');
        const failed = ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(job.status);
        const statusClass = job.status === 'COMPLETED' ? 'chip-done' : failed ? 'chip-error' :
          ['RUNNING', 'STOPPING'].includes(job.status) ? 'chip-partial' : 'chip-pending';
        row.innerHTML = `<td>${esc(job.class_name)}/${esc(job.model_id)}</td><td>${esc(actionLabel(job.action))}${job.action.includes('evaluate') ? ' · ' + esc(job.evaluation_method) : ''}</td>
          <td><span class="chip ${statusClass}">${esc(job.status)}</span></td>
          <td id="job-duration-${esc(job.id)}">${job.started_at ? fmtSec((job.finished_at || Date.now() / 1000) - job.started_at) : '-'}</td><td></td>`;
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
      if (!jobs.length) $('queueBody').innerHTML = '<tr><td colspan="5" class="empty-msg"><strong>Antrean masih kosong</strong>Pilih model di bawah untuk memulai rekonstruksi pertama.</td></tr>';
      lastQueueSignature = queueSignature;
    }
    for (const job of jobs) {
      const duration = $(`job-duration-${job.id}`);
      if (duration) duration.textContent = job.started_at ? fmtSec((job.finished_at || Date.now() / 1000) - job.started_at) : '-';
    }
    if (data.logs.length) renderLogs(data.logs);
    if (selectedLogId) await refreshSelectedJobLogs();
    const signature = data.jobs.map(j => j.id + j.status).join();
    if (signature !== lastJobId) { lastJobId = signature; refreshStatus(false); refreshMetrics(); }
  } catch (error) { console.error(error); }
}

async function showJobLogs(id) {
  selectedLogId = id;
  selectedLogRequest++;
  $('jobLogTitle').textContent = 'Log pekerjaan';
  $('jobLogStatus').textContent = 'Memuat...';
  $('jobLogContent').textContent = 'Memuat log...';
  if (!$('jobLogDialog').open) $('jobLogDialog').showModal();
  await refreshSelectedJobLogs(true);
}

async function refreshSelectedJobLogs(force = false) {
  if (!selectedLogId || (selectedLogLoading && !force)) return;
  const id = selectedLogId;
  const request = selectedLogRequest;
  selectedLogLoading = true;
  try {
    const data = await api(`/api/jobs/${id}/logs`);
    if (selectedLogId !== id || request !== selectedLogRequest) return;
    $('jobLogTitle').textContent = `${data.job.class_name}/${data.job.model_id} · ${data.job.action}`;
    $('jobLogStatus').textContent = `${data.job.status} · diperbarui otomatis setiap 2 detik · 300 baris terakhir`;
    $('jobLogContent').textContent = data.logs.length ? data.logs.join('\n') : 'Belum ada log untuk pekerjaan ini.';
    if ($('jobLogFollow').checked) $('jobLogContent').scrollTop = $('jobLogContent').scrollHeight;
  } catch (error) {
    if (selectedLogId === id && request === selectedLogRequest) {
      $('jobLogStatus').textContent = 'Gagal memuat log';
      $('jobLogContent').textContent = error.message;
    }
  } finally {
    if (selectedLogId === id && request === selectedLogRequest) selectedLogLoading = false;
  }
}
function closeJobLogs() { selectedLogId = null; selectedLogRequest++; selectedLogLoading = false; $('jobLogDialog').close(); }
function showLiveLogs() { closeJobLogs(); $('termLabel').textContent = 'output'; pollLogs(); }

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
      const evaluated = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)', 'Precision @2cm (%)', 'Precision @5cm (%)'].some(column => Number.isFinite(parseFloat(row[column])));
      for (const [column, value] of Object.entries(row)) {
        if (evaluated && ['Total Waktu (Menit)', 'Peak VRAM (GB)'].includes(column)) continue;
        if (value !== '-' && value !== '' && value != null) current[column] = value;
      }
      latest.set(key, current);
    }
    cachedMetrics = [...latest.values()];
    renderMetrics();
  } catch (error) { console.error(error); }
}

function renderMetrics() {
  const method = $('metricMethod').value || 'all';
  const threshold = $('geometryThreshold').value === '5' ? 5 : 2;
  const display = cachedMetrics.filter(row => method === 'all' || row.Metode === method);
  const precision = `Precision @${threshold}cm (%)`, recall = `Recall @${threshold}cm (%)`, f1 = `F1 @${threshold}cm (%)`;
  const columns = ['Timestamp', 'Kelas', 'Model ID', 'Metode', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)',
    precision, recall, f1, 'Peak VRAM (GB)', 'Total Waktu (Menit)'];
  $('metricsBody').innerHTML = display.map(row => '<tr>' + columns.map((column, i) => `<td>${esc(i < 4 ? row[column] || '-' : fmtN(row[column]))}</td>`).join('') + '</tr>').join('') || '<tr><td colspan="13" class="empty-msg"><strong>Belum ada hasil evaluasi</strong>Jalankan evaluasi visual; metrik geometri memerlukan ground truth LiDAR dan kalibrasi skala.</td></tr>';
  for (const [column, target, context, unit, digits, explanation] of [
    ['PSNR (dB)', 'kpiPsnr', 'psnrContext', ' dB', 2, '↑'],
    ['SSIM', 'kpiSsim', 'ssimContext', '', 4, '↑'],
    ['LPIPS', 'kpiLpips', 'lpipsContext', '', 4, '↓'],
    ['Chamfer Dist (cm)', 'kpiChamfer', 'chamferContext', ' cm', 2, '↓'],
    [precision, 'kpiPrecision', 'precisionContext', '%', 2, `ambang ${threshold} cm · ↑`],
    [recall, 'kpiRecall', 'recallContext', '%', 2, `ambang ${threshold} cm · ↑`],
    [f1, 'kpiF1', 'f1Context', '%', 2, `ambang ${threshold} cm · ↑`]]) {
    const values = display.map(row => Number.parseFloat(row[column])).filter(Number.isFinite);
    $(target).textContent = values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(digits) + unit : '—';
    $(context).textContent = `${values.length ? 'Rata-rata ' + values.length + ' hasil' : 'Belum dievaluasi'} · ${explanation}`;
  }
  $('metricScope').textContent = `Rata-rata hasil terakhir per model/metode · ${method === 'all' ? '3DGS + SuGaR' : method}. Model selesai menghitung pipeline kedua metode. Geometri memerlukan LiDAR.`;
  for (const [id, label] of [['precisionHeader', 'Precision'], ['recallHeader', 'Recall'], ['f1Header', 'F1']]) {
    $(id).textContent = `${label} @${threshold}cm (%)`;
  }
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
    if (!data.images.length) {
      box.innerHTML = '<span class="preview-empty">Preview render tersedia setelah evaluasi</span>';
      return;
    }
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
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') { closeLightbox(); closeJobLogs(); }
});
