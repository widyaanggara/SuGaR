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
let metricExperiment = 'baseline';
const metricMethods = {baseline: ['3DGS', 'SuGaR'], scaffold_anchor: ['3DGS_scaffold_anchor', 'SuGaR_scaffold_anchor']};
let editingModel = null;
let modelNameSaving = false;
let queueSelection = null;
let selectedQueueAction = 'full_pipeline';
let queueSubmitting = false;
let calibrationSaving = false;
const geometryInputs = new Map();
const queueActions = ['full_pipeline', 'colmap', '3dgs', 'sugar', 'evaluate_visual', 'evaluate', 'extract_frames'];

const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtN = value => Number.isFinite(parseFloat(value)) ? parseFloat(value).toFixed(4) : '-';
const fmtSec = seconds => [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, Math.floor(seconds) % 60].map(n => String(n).padStart(2, '0')).join(':');
const actionLabel = action => ({full_pipeline: 'Pipeline lengkap', colmap: 'Rekonstruksi COLMAP',
  '3dgs': 'Training 3DGS', sugar: 'Training SuGaR', extract_frames: 'Ekstraksi frame',
  evaluate_visual: 'Evaluasi visual', evaluate: 'Evaluasi geometri'}[action] || action || '');
const settingsTabs = ['resolution', 'colmap', 'training', 'lidar'];
const isFilePreview = window.location?.protocol === 'file:' || window.location?.search === '?preview=1';
const previewMessage = 'Pratinjau desain lokal. Data penelitian dan eksekusi pipeline tersedia melalui dashboard yang berjalan di server.';

function defaultModelName(mdl) {
  return mdl.startsWith('model_') ? 'Model ' + mdl.slice(6) : 'Benchmark ' + mdl.split('_').at(-1);
}

function modelName(cls, mdl) {
  return cachedStatus[cls]?.[mdl]?.display_name || defaultModelName(mdl);
}

function modelIdentityHtml(cls, mdl, displayName = modelName(cls, mdl)) {
  return `<span>${esc(displayName)}</span><small class="model-reference">${esc(cls)}/${esc(mdl)}</small>`;
}

function editModelName(cls, mdl) {
  if (modelNameSaving || !cachedStatus[cls]?.[mdl]) return;
  editingModel = {class_name: cls, model_id: mdl};
  $('modelNameTarget').textContent = `${cls} / ${mdl}`;
  $('modelNameInput').value = cachedStatus[cls][mdl].custom_name || '';
  $('modelNameInput').placeholder = defaultModelName(mdl);
  $('modelNameError').textContent = '';
  $('modelNameDialog').showModal();
  $('modelNameInput').focus();
}

function closeModelName() {
  if (!modelNameSaving) {
    $('modelNameDialog').close();
    editingModel = null;
  }
}

async function saveModelName() {
  if (!editingModel || modelNameSaving) return;
  const target = editingModel;
  modelNameSaving = true;
  $('modelNameSave').disabled = true;
  $('modelNameCancel').disabled = true;
  $('modelNameInput').disabled = true;
  $('modelNameError').textContent = '';
  try {
    const result = await api('/api/model-name', {...target, display_name: $('modelNameInput').value.trim()});
    const info = cachedStatus[target.class_name]?.[target.model_id];
    if (info) {
      info.display_name = result.display_name;
      info.custom_name = result.custom_name;
    }
    renderGrid(cachedStatus);
    renderMetrics();
    $('modelNameDialog').close();
    editingModel = null;
    showToast(`Nama disimpan: ${result.display_name}`);
    await pollLogs();
  } catch (error) {
    $('modelNameError').textContent = error.message;
  } finally {
    modelNameSaving = false;
    $('modelNameSave').disabled = false;
    $('modelNameCancel').disabled = false;
    $('modelNameInput').disabled = false;
  }
}

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
  const compact = window.matchMedia?.('(max-width: 850px)');
  const orientTabs = () => $('settingsTabList').setAttribute('aria-orientation', compact?.matches ? 'horizontal' : 'vertical');
  orientTabs();
  compact?.addEventListener?.('change', orientTabs);
  for (const [index, name] of settingsTabs.entries()) {
    $('tab-' + name).onkeydown = event => {
      let next;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % settingsTabs.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index + settingsTabs.length - 1) % settingsTabs.length;
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
  if (isFilePreview) throw new Error(previewMessage);
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
  'gaussiansPerTriangle', 'regularization', 'evaluationMethod'];
let presets = {};
const sugarControlIds = {vertices: 'meshTriangles', gaussians_per_triangle: 'gaussiansPerTriangle',
  refinement_time: 'refinementTime', regularization: 'regularization'};

function readExperimentSettings() {
  return {
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
    evaluation_method: $('evaluationMethod').value
  };
}

function renderSettingsSummary() {
  const values = readExperimentSettings();
  const device = {auto: 'Otomatis', cpu: 'CPU', cuda: 'CUDA'}[values.colmap_device];
  const validSize = value => Number.isInteger(value) && value >= 256 && value <= 16384;
  $('summaryGpu').textContent = Number.isInteger(values.gpu) && values.gpu >= 0 ? `GPU ${values.gpu}` : 'Periksa GPU ID';
  $('summaryColmap').textContent = device && validSize(values.colmap_max_image_size) ? `${device} / maks. ${values.colmap_max_image_size} px` : 'Periksa pengaturan COLMAP';
  $('summaryTraining').textContent = [1, 2, 4, 8].includes(values.image_downscale) && (values.image_max_size === 0 || validSize(values.image_max_size)) ?
    `${values.image_downscale}× / ${values.image_max_size ? 'maks. ' + values.image_max_size + ' px' : 'tanpa batas tambahan'}` : 'Periksa resolusi training';
  const refinement = {short: '2.000', medium: '7.000', long: '15.000'}[values.refinement_time];
  $('summaryRefinement').textContent = refinement ? `${refinement} iterasi` : 'Pilih refinement';
}

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
  renderSettingsSummary();
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
  if (isFilePreview) {
    // Empty slots are design fixtures, never presented as server research data.
    cachedStatus = Object.fromEntries(['Dewa', 'Dewi', 'Mitologi', 'Penabuh', 'Pengapit', 'Punakawan', 'Raksasa', 'Wanara', 'TanksAndTemples']
      .map(cls => [cls, Object.fromEntries((cls === 'TanksAndTemples' ? ['tanks_and_temples_1', 'tanks_and_temples_2'] : ['model_01', 'model_02'])
        .map(mdl => [mdl, {images_count: 0, input_path: `datasets/${cls}/${mdl}/input/`, output_path: `outputs/{3dgs,sugar}/${cls}/${mdl}/`}]))]));
    initializeSettings();
    $('previewNotice').hidden = false;
    $('previewNotice').textContent = previewMessage;
    $('settingsPreset').disabled = true;
    $('resetSugarPreset').disabled = true;
    $('presetDescription').textContent = 'Pratinjau pengaturan kustom. Preset resmi dimuat melalui dashboard server; perubahan di sini tidak mengubah pekerjaan penelitian.';
    $('btnCheckColmap').disabled = true;
    $('colmapCapability').textContent = 'Pemeriksaan build COLMAP tersedia di server.';
    $('queueSummary').textContent = 'Pratinjau lokal — data antrean server tidak dimuat';
    $('queueBody').innerHTML = '<tr><td colspan="5" class="empty-msg">Data antrean tersedia pada dashboard server.</td></tr>';
    $('kpiCompleted').textContent = '— / 18';
    $('kpiCompletedSub').textContent = 'Data server tidak dimuat di pratinjau';
    renderMetrics();
    renderGrid(cachedStatus);
    return;
  }
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
  if (isFilePreview) { showToast(previewMessage); return; }
  lastGridSignature = null;
  await Promise.all([refreshStatus(false), refreshMetrics(), pollLogs()]);
}

async function refreshStatus(feedback) {
  try {
    cachedStatus = await api('/api/status');
    renderGrid(cachedStatus);
    renderMetrics();
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
      const modelLabel = info.display_name || defaultModelName(mdl);
      cells += `<div class="model-cell">
        <div class="model-label"><div class="model-identity"><span class="model-title">${esc(modelLabel)}</span><span class="model-code">${esc(mdl)}</span></div><div class="model-tools">${info.has_video ? chip(true, 'Video') : ''}<button type="button" class="btn-name" aria-label="Ubah nama ${esc(cls)} ${esc(mdl)}" onclick="editModelName(${target})">Ubah nama</button></div></div>
        <div class="pipeline-rows">
          <div class="pipeline-row"><span><span class="stage-index">01</span>Foto input</span>${chip(info.images_count > 0, `${info.images_count} foto`, 'Belum ada foto')}</div>
          <div class="pipeline-row"><span><span class="stage-index">02</span>COLMAP</span>${chip(info.colmap_done)}</div>
          <div class="pipeline-row"><span><span class="stage-index">03</span>3DGS</span>${chip(info['3dgs_30k'], '30k iter', info['3dgs_7k'] ? '7k iter' : 'Menunggu')}</div>
          <div class="pipeline-row"><span><span class="stage-index">04</span>SuGaR + OBJ</span>${chip(info.sugar_obj, 'OBJ lengkap', info.sugar_mesh ? 'Mesh kasar' : 'Menunggu')}</div>
          <div class="pipeline-row"><span><span class="stage-index">05</span>Evaluasi visual</span>${chip(info.evaluated)}</div>
          <div class="pipeline-row"><span><span class="stage-index">06</span>Evaluasi geometri</span>${chip(info.geometry_evaluated, '2 metode selesai', !info.has_gt_lidar ? 'Perlu LiDAR' : !info.geometry_settings ? 'Atur LiDAR' : 'Menunggu')}</div>
        </div>
        <details class="dataset-path"><summary>Lihat folder input &amp; output</summary><code>${esc(info.input_path)}</code><code>${esc(info.output_path)}</code><code>${esc(`datasets/${cls}/${mdl}/ground_truth/lidar.ply`)}</code></details>
        <div class="step-buttons">
          <button class="btn btn-primary btn-sm btn-full" onclick="openQueueDialog(${target})">Antrekan model ini</button>
        </div><div class="preview-box" id="preview-box-${cls}-${mdl}"></div></div>`;
    }
    const number = String(Object.keys(data).indexOf(cls) + 1).padStart(2, '0');
    card.innerHTML = `<div class="class-header"><div class="class-heading"><span class="class-number">${number}</span><div><div class="class-name">${esc(cls === 'TanksAndTemples' ? 'Tanks & Temples' : cls)}</div><div class="class-meta">${Object.keys(models).length} model · ${cls === 'TanksAndTemples' ? 'Benchmark rekonstruksi' : 'Tipologi Togog Bali'}</div></div></div>
      <button class="btn btn-ghost btn-sm" onclick="openQueueDialog('${cls}','')">Antrekan Kelas</button></div>
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

function selectedModelPairs() {
  if (!queueSelection) return [];
  return Object.entries(cachedStatus).flatMap(([cls, models]) => Object.entries(models)
    .filter(([mdl]) => (!queueSelection.class_name || queueSelection.class_name === cls) &&
      (!queueSelection.model_id || queueSelection.model_id === mdl))
    .map(([mdl, info]) => ({cls, mdl, info})));
}

function openQueueDialog(cls = '', mdl = '', action = 'full_pipeline') {
  if (queueSubmitting || calibrationSaving) return;
  queueSelection = {class_name: cls, model_id: mdl};
  $('queueClass').innerHTML = '<option value="">Semua kelas</option>' + Object.keys(cachedStatus)
    .map(name => `<option value="${esc(name)}">${esc(name)}</option>`).join('');
  $('queueClass').value = cls;
  populateQueueModels(mdl);
  $('queueMethod').value = $('evaluationMethod').value || 'both';
  $('queuePipelineStart').value = 'auto';
  $('queueApplyIcp').checked = true;
  $('queueAlignment').value = mdl ? cachedStatus[cls]?.[mdl]?.geometry_settings?.alignment_mode || 'automatic' : 'automatic';
  $('queueDialogError').textContent = '';
  geometryInputs.clear();
  renderQueueTargets();
  chooseQueueAction(action);
  $('queueDialog').showModal();
}

function populateQueueModels(mdl = '') {
  const cls = $('queueClass').value;
  $('queueModel').innerHTML = '<option value="">Semua model</option>' + (cls ? Object.keys(cachedStatus[cls] || {})
    .map(id => `<option value="${esc(id)}">${esc(modelName(cls, id))} (${esc(id)})</option>`).join('') : '');
  $('queueModel').value = mdl;
  $('queueModel').disabled = !cls;
}

function changeQueueTarget(classChanged) {
  if (queueSubmitting || calibrationSaving) return;
  if (classChanged) populateQueueModels();
  queueSelection = {class_name: $('queueClass').value, model_id: $('queueModel').value};
  renderQueueTargets();
  const single = !!queueSelection.class_name && !!queueSelection.model_id;
  chooseQueueAction(!single && !['full_pipeline', 'evaluate_visual', 'evaluate'].includes(selectedQueueAction)
    ? 'full_pipeline' : selectedQueueAction);
}

function renderQueueTargets() {
  const {class_name: cls, model_id: mdl} = queueSelection;
  const pairs = selectedModelPairs();
  $('queueDialogHeading').textContent = mdl ? 'Antrekan model ini' : 'Antrekan model';
  $('queueDialogScope').textContent = mdl ? `${modelName(cls, mdl)} · ${cls}/${mdl}` :
    cls ? `Pilih proses untuk kelas ${cls}.` : 'Atur target dan proses, lalu tambahkan ke antrean.';
  $('queueTargetCount').textContent = `${pairs.length} model dipilih`;
  $('geometryCalibrationBody').innerHTML = '';
  for (const {cls, mdl, info} of pairs) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.innerHTML = `${modelIdentityHtml(cls, mdl)}<div class="geometry-file-state">${chip(info.has_gt_lidar, 'LiDAR ada', 'LiDAR belum ada')}${chip(info.geometry_sources?.['3DGS'], '3DGS ada', '3DGS belum ada')}${chip(info.geometry_sources?.SuGaR, 'SuGaR ada', 'SuGaR belum ada')}</div>`;
    row.appendChild(cell);
    let inputs = geometryInputs.get(`${cls}/${mdl}`);
    if (!inputs) {
      const unit = document.createElement('select');
      unit.innerHTML = '<option value="">Pilih satuan…</option><option value="m">Meter (m)</option><option value="cm">Sentimeter (cm)</option><option value="mm">Milimeter (mm)</option>';
      unit.setAttribute('aria-label', `Satuan LiDAR ${cls} ${mdl}`);
      unit.value = info.geometry_settings?.lidar_unit || '';
      unit.onchange = updateQueuePlan;
      const scale = document.createElement('input');
      scale.type = 'number'; scale.min = '0'; scale.step = 'any'; scale.placeholder = 'Hasil kalibrasi';
      scale.setAttribute('aria-label', `Skala manual ${cls} ${mdl}`);
      scale.value = String(info.geometry_settings?.scale_factor ?? info.geometry_scale_factor ?? '');
      scale.oninput = updateQueuePlan;
      inputs = {unit, scale};
      geometryInputs.set(`${cls}/${mdl}`, inputs);
    }
    for (const input of [inputs.unit, inputs.scale]) {
      const inputCell = document.createElement('td');
      inputCell.appendChild(input);
      if (input === inputs.scale) inputs.scaleCell = inputCell;
      row.appendChild(inputCell);
    }
    $('geometryCalibrationBody').appendChild(row);
  }
}

function closeQueueDialog() {
  if (queueSubmitting || calibrationSaving) return;
  $('queueDialog').close();
  queueSelection = null;
  geometryInputs.clear();
}

function chooseQueueAction(action) {
  if (!queueActions.includes(action) || queueSubmitting || calibrationSaving) return;
  const single = !!queueSelection?.class_name && !!queueSelection?.model_id;
  $('queueProcessOptions').dataset.scope = single ? 'single' : 'bulk';
  if (!single && !['full_pipeline', 'evaluate_visual', 'evaluate'].includes(action)) return;
  selectedQueueAction = action;
  for (const name of queueActions) {
    const button = $('queue-action-' + name);
    button.setAttribute('aria-pressed', String(name === action));
    button.disabled = !single && !['full_pipeline', 'evaluate_visual', 'evaluate'].includes(name);
    button.hidden = !single && !['full_pipeline', 'evaluate_visual', 'evaluate'].includes(name);
  }
  $('queue-action-extract_frames').hidden = !single || !selectedModelPairs()[0]?.info.has_video;
  const evaluation = ['evaluate_visual', 'evaluate'].includes(action);
  $('queueMethodOptions').hidden = !evaluation;
  $('queueGeometryOptions').hidden = action !== 'evaluate';
  $('queuePipelineOptions').hidden = action !== 'full_pipeline';
  $('queueActionHelp').textContent = {
    full_pipeline: 'Mengikuti pengaturan eksperimen hingga evaluasi visual. Evaluasi geometri dijalankan terpisah setelah scan LiDAR tersedia.',
    colmap: 'Menghitung fitur, pencocokan, pose kamera, dan undistorsi dari foto input.',
    '3dgs': 'Melatih baseline 3DGS dari dataset COLMAP yang sudah tersedia.',
    sugar: 'Menjalankan SuGaR menggunakan checkpoint 3DGS 7.000 iterasi yang sudah tersedia.',
    evaluate_visual: 'Menghitung PSNR, SSIM, dan LPIPS dari checkpoint yang ada. Tidak menjalankan training ulang.',
    evaluate: 'Menghitung Chamfer, Precision, Recall, dan F1 bersama pada ambang 2 cm dan 5 cm, tanpa training ulang.',
    extract_frames: 'Mengekstrak frame video ke input. Folder input harus belum berisi foto.'
  }[action];
  $('queueDialogError').textContent = '';
  updateQueuePlan();
}

function queuePlan() {
  const action = selectedQueueAction;
  const evaluation = ['evaluate_visual', 'evaluate'].includes(action);
  const methods = evaluation ? ($('queueMethod').value === 'both' ? ['3DGS', 'SuGaR'] : [$('queueMethod').value]) : [''];
  const ready = [], skipped = [];
  for (const {cls, mdl, info} of selectedModelPairs()) {
    for (const method of methods) {
      let reason = '';
      if (action === 'full_pipeline' && !info.images_count && !info.has_video) reason = 'foto/video belum tersedia';
      if (action === 'full_pipeline' && $('queuePipelineStart').value === 'existing_colmap' && !info.colmap_done) reason = 'images/ hasil undistorsi dan sparse/0 COLMAP belum lengkap';
      if (action === 'colmap' && !info.images_count) reason = 'foto input belum tersedia';
      if (action === '3dgs' && !info.colmap_done) reason = 'hasil COLMAP belum lengkap';
      if (action === 'sugar' && (!info.colmap_done || !info['3dgs_7k'])) reason = 'COLMAP dan checkpoint 3DGS 7k diperlukan';
      if (action === 'extract_frames' && (!info.has_video || info.input_count > 0)) reason = 'perlu video dan folder input kosong';
      if (action === 'evaluate_visual' && !(method === '3DGS' ? info['3dgs_30k'] : info['3dgs_7k'] && info.sugar_obj)) reason = `checkpoint ${method} belum lengkap`;
      if (action === 'evaluate') {
        const inputs = geometryInputs.get(`${cls}/${mdl}`);
        const text = inputs?.scale.value.trim() || '';
        if (!info.has_gt_lidar) reason = 'ground_truth/lidar.ply belum tersedia';
        else if (!['m', 'cm', 'mm'].includes(inputs?.unit.value)) reason = 'pilih satuan koordinat LiDAR';
        else if ($('queueAlignment').value === 'manual' && (!text || !Number.isFinite(Number(text)) || Number(text) <= 0)) reason = 'isi faktor skala manual positif';
        else if (!info.geometry_sources?.[method]) reason = `hasil rekonstruksi ${method} belum tersedia`;
      }
      const target = `${modelName(cls, mdl)} (${cls}/${mdl}${method ? ' · ' + method : ''})`;
      if (reason) skipped.push(`${target}: ${reason}`);
      else ready.push({cls, mdl, method});
    }
  }
  return {ready, skipped};
}

function updateQueuePlan() {
  if (!queueSelection) return;
  const {ready, skipped} = queuePlan();
  const busy = queueSubmitting || calibrationSaving;
  const manual = $('queueAlignment').value === 'manual';
  const selectedInputs = selectedModelPairs().map(({cls, mdl}) => geometryInputs.get(`${cls}/${mdl}`));
  const invalidScale = selectedQueueAction === 'evaluate' && manual && selectedInputs.some(inputs => {
    const text = inputs.scale.value.trim();
    return text && (!Number.isFinite(Number(text)) || Number(text) <= 0);
  });
  $('queuePlanSummary').textContent = isFilePreview ? 'Pratinjau lokal — tidak mengirim pekerjaan' : invalidScale ? 'Perbaiki skala manual yang tidak valid' : `${ready.length} pekerjaan siap diantrekan`;
  $('queuePlanNote').textContent = skipped.length ? `${skipped.length} target belum siap dan akan dilewati. Buka rincian untuk melihat yang perlu disiapkan.` : 'Semua target siap. Pekerjaan berjalan bergantian menggunakan pengaturan eksperimen saat ini.';
  $('queueSkippedDetails').hidden = !skipped.length;
  $('queueSkippedSummary').textContent = `${skipped.length} target yang belum siap`;
  $('queueSkippedList').innerHTML = skipped.map(reason => `<li>${esc(reason)}</li>`).join('');
  $('queueDialogSubmit').disabled = isFilePreview || busy || invalidScale || !ready.length;
  $('queueDialogSubmit').textContent = queueSubmitting ? 'Menambahkan...' : `Antrekan ${ready.length} pekerjaan`;
  $('queueDialogCancel').disabled = busy;
  $('queueDialogClose').disabled = busy;
  $('queueSaveCalibration').disabled = isFilePreview || busy || invalidScale || !selectedInputs.some(inputs => inputs.unit.value && (!manual || Number(inputs.scale.value) > 0));
  $('queueMethod').disabled = busy;
  $('queuePipelineStart').disabled = busy;
  const scaffoldSelected = $('queueExperiment').value === 'scaffold_anchor';
  $('queueScaffoldOptions').hidden = !scaffoldSelected;
  $('queueExperiment').disabled = busy;
  for (const id of ['queueScaffoldRepo', 'queueScaffoldPython', 'queueScaffoldIterations', 'queueScaffoldResolution']) $(id).disabled = busy;
  $('queuePipelineHelp').textContent = $('queuePipelineStart').value === 'existing_colmap'
    ? 'Memakai images/ dan sparse/0 yang sudah ada, termasuk hasil dari komputer lain. Langsung menjalankan 3DGS → SuGaR → evaluasi visual. COLMAP tidak dijalankan ulang; target yang belum lengkap dilewati. Pengaturan resolusi training tetap berlaku.'
    : 'Dengan Lewati tahap yang selesai aktif, setiap model melanjutkan dari hasil terakhir yang tersedia. COLMAP lengkap, termasuk hasil teman, langsung dipakai. Training dan evaluasi yang lengkap serta cocok dilewati; checkpoint parsial yang cocok dipulihkan. Pengaturan COLMAP diterapkan hanya saat membuat COLMAP baru. Untuk interrupted, gunakan Antrekan Ulang agar pengaturan asli dan prioritas pemulihannya dipertahankan.';
  $('queueApplyIcp').disabled = busy;
  if (scaffoldSelected) $('queuePipelineHelp').textContent = 'COLMAP → Scaffold-GS → anchor dengan warna COLMAP → 3DGS → SuGaR → evaluasi. Output ablasi terpisah. Tahap Scaffold yang terputus dimulai ulang; tahap Scaffold yang selesai dan cocok dilewati.';
  $('queueAlignment').disabled = busy;
  $('queueClass').disabled = busy;
  $('queueModel').disabled = busy || !queueSelection.class_name;
  $('geometryScaleHeading').hidden = !manual;
  $('queueIcpOption').hidden = !manual;
  $('queueAlignmentNote').textContent = manual ? 'Faktor manual mengubah unit rekonstruksi menjadi meter. Satuan file LiDAR tetap harus dipilih dengan benar.' : 'Tidak perlu memasukkan tinggi atau faktor skala. Sistem mencari penyelarasan, menyimpan transformasi dan model overlay, lalu menghentikan evaluasi jika pemeriksaan numeriknya gagal. Tetap periksa overlay untuk skripsi.';
  for (const inputs of selectedInputs) {
    inputs.unit.disabled = busy;
    inputs.scale.disabled = busy;
    inputs.scaleCell.hidden = !manual;
  }
  const single = !!queueSelection.class_name && !!queueSelection.model_id;
  for (const action of queueActions) $('queue-action-' + action).disabled = busy ||
    (!single && !['full_pipeline', 'evaluate_visual', 'evaluate'].includes(action));
}

async function persistQueueCalibration() {
  const mode = $('queueAlignment').value;
  const entries = selectedModelPairs().flatMap(({cls, mdl}) => {
    const inputs = geometryInputs.get(`${cls}/${mdl}`);
    const text = inputs.scale.value.trim();
    if (!inputs.unit.value || (mode === 'manual' && !text)) return [{class_name: cls, model_id: mdl, clear: true}];
    if (mode === 'manual' && (!Number.isFinite(Number(text)) || Number(text) <= 0)) throw new Error(`Skala ${cls}/${mdl} harus positif.`);
    return [{class_name: cls, model_id: mdl, lidar_unit: inputs.unit.value, alignment_mode: mode,
      ...(mode === 'manual' ? {scale_factor: Number(text)} : {})}];
  });
  if (!entries.some(entry => !entry.clear)) throw new Error('Pilih satuan LiDAR untuk setidaknya satu model.');
  const result = await api('/api/geometry-settings', {settings: entries});
  for (const {cls, mdl, info} of selectedModelPairs()) info.geometry_settings = result.settings[cls]?.[mdl] ?? null;
  return result;
}

async function saveQueueCalibration() {
  if (queueSubmitting || calibrationSaving || !queueSelection || isFilePreview) return;
  calibrationSaving = true;
  updateQueuePlan();
  try {
    await persistQueueCalibration();
    $('queueDialogError').textContent = '';
    showToast('Pengaturan LiDAR per model disimpan di server.');
  } catch (error) { $('queueDialogError').textContent = error.message; }
  finally { calibrationSaving = false; updateQueuePlan(); }
}

async function submitQueueDialog() {
  if (!queueSelection || queueSubmitting || calibrationSaving || isFilePreview) return;
  if (!queuePlan().ready.length) { $('queueDialogError').textContent = 'Belum ada model yang siap untuk proses ini.'; return; }
  const action = selectedQueueAction;
  const payload = {...readExperimentSettings(), ...queueSelection, action,
    pipeline_start: action === 'full_pipeline' ? $('queuePipelineStart').value : 'auto',
    experiment: action === 'full_pipeline' ? $('queueExperiment').value : 'baseline',
    scaffold_repo: $('queueScaffoldRepo').value.trim(),
    scaffold_python: $('queueScaffoldPython').value.trim(),
    scaffold_iterations: Number($('queueScaffoldIterations').value),
    scaffold_resolution: Number($('queueScaffoldResolution').value),
    evaluation_method: ['evaluate_visual', 'evaluate'].includes(action) ? $('queueMethod').value : 'SuGaR',
    geometry_settings_required: action === 'evaluate',
    apply_icp: $('queueAlignment').value === 'automatic' || $('queueApplyIcp').checked};
  queueSubmitting = true;
  updateQueuePlan();
  $('queueDialogError').textContent = '';
  try {
    if (action === 'evaluate') await persistQueueCalibration();
    const result = await api('/api/run', payload);
    const skipped = result.skipped || result.skipped_empty || [];
    showToast(`${result.added.length} pekerjaan ditambahkan; ${result.duplicates.length} sudah di antrean; ${skipped.length} target dilewati.`);
    if (result.added.length || result.duplicates.length) {
      $('queueDialog').close();
      queueSelection = null;
      geometryInputs.clear();
    } else $('queueDialogError').textContent = skipped.join('\n') || 'Belum ada pekerjaan yang dapat ditambahkan.';
    await pollLogs();
  } catch (error) { $('queueDialogError').textContent = error.message; }
  finally { queueSubmitting = false; updateQueuePlan(); }
}

async function triggerRun(action, cls, mdl) {
  if (action === 'evaluate') { openQueueDialog(cls, mdl, action); return; }
  const payload = {action, class_name: cls, model_id: mdl, ...readExperimentSettings()};
  if (!['evaluate', 'evaluate_visual'].includes(action)) payload.evaluation_method = 'SuGaR';
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
    if (queueSelection) updateQueuePlan();
    $('btnStop').disabled = !data.is_running;
    $('jobStrip').classList.toggle('hidden', !data.is_running);
    $('jobLabel').textContent = actionLabel(data.action);
    const activeJob = data.jobs.find(job => ['RUNNING', 'STOPPING'].includes(job.status));
    $('jobTarget').textContent = activeJob ? `${modelName(activeJob.class_name, activeJob.model_id)} · ${activeJob.class_name}/${activeJob.model_id}` : data.target || '';
    $('jobTimer').textContent = fmtSec(data.elapsed_seconds);
    $('termStatus').textContent = data.status.toLowerCase();
    $('termStatus').className = 'chip ' + (data.status === 'COMPLETED' ? 'chip-done' : data.status === 'FAILED' ? 'chip-warn' : 'chip-partial');
    $('queueSummary').textContent = `${data.paused ? 'Dijeda' : 'Aktif'} · ${data.counts.RUNNING || 0} berjalan · ${data.counts.QUEUED || 0} menunggu · ${data.counts.COMPLETED || 0} selesai · ${data.counts.FAILED || 0} gagal`;
    $('btnPause').textContent = data.paused ? 'Lanjutkan Antrean' : 'Jeda Antrean';
    $('btnPause').onclick = () => queueControl(data.paused ? 'resume' : 'pause');
    const waiting = data.jobs.filter(j => ['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status));
    waiting.sort((a, b) => {
      if (Number.isFinite(a.queue_position) && Number.isFinite(b.queue_position)) return a.queue_position - b.queue_position;
      const rank = job => ['RUNNING', 'STOPPING'].includes(job.status) ? 0 : job.recovery ? 1 : 2;
      return rank(a) - rank(b) || (a.recovery_requested_at ?? a.created_at ?? 0) - (b.recovery_requested_at ?? b.created_at ?? 0);
    });
    const jobs = [...waiting,
                  ...data.jobs.filter(j => !['RUNNING', 'STOPPING', 'QUEUED'].includes(j.status)).reverse().slice(0, 30)];
    const queueSignature = JSON.stringify(jobs.map(job => [job.id, job.status, job.error, job.recovery, modelName(job.class_name, job.model_id)]));
    // Keep buttons in the DOM while the timer ticks, so polling cannot swallow a click.
    if (queueSignature !== lastQueueSignature) {
      $('queueBody').innerHTML = '';
      for (const job of jobs) {
        const row = document.createElement('tr');
        const failed = ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(job.status);
        const statusClass = job.status === 'COMPLETED' ? 'chip-done' : failed ? 'chip-error' :
          ['RUNNING', 'STOPPING'].includes(job.status) ? 'chip-partial' : 'chip-pending';
        row.innerHTML = `<td>${modelIdentityHtml(job.class_name, job.model_id)}</td><td>${esc(actionLabel(job.action))}${job.action.includes('evaluate') ? ' · ' + esc(job.evaluation_method) : ''}${job.experiment === 'scaffold_anchor' ? '<small class="model-reference">Ablasi anchor Scaffold-GS</small>' : ''}${job.pipeline_start === 'existing_colmap' ? '<small class="model-reference">Mulai dari hasil COLMAP</small>' : ''}${job.recovery ? '<small class="model-reference">Prioritas pemulihan checkpoint</small>' : ''}</td>
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
          if (failed) button.title = 'Pulihkan checkpoint yang cocok dan prioritaskan setelah pekerjaan aktif selesai.';
          button.onclick = failed ? async () => {
            try {
              const result = await api('/api/queue/retry', {job_id: job.id});
              showToast(result.already_running ? 'Model ini sudah berjalan.' : 'Pemulihan diprioritaskan setelah pekerjaan aktif selesai. Jika antrean dijeda, klik Lanjutkan Antrean.');
              await pollLogs();
            } catch (error) { showToast(error.message); }
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
    $('jobLogTitle').textContent = `${modelName(data.job.class_name, data.job.model_id)} · ${data.job.class_name}/${data.job.model_id} · ${data.job.action}`;
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
    const rows = (await api('/api/metrics')).filter(row => Object.values(metricMethods).flat().includes(row.Metode));
    const latest = new Map();
    for (const row of rows.sort((a, b) => (a.Timestamp || '').localeCompare(b.Timestamp || ''))) {
      if (row.Status !== 'COMPLETED') continue;
      const key = [row.Kelas, row['Model ID'], row.Metode].join('/');
      const current = latest.get(key) || {};
      // Training rows, visual evaluation and geometry evaluation stay linked by model/method.
      const evaluated = ['PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)', 'Precision @2cm (%)', 'Precision @5cm (%)', 'Recall @2cm (%)', 'Recall @5cm (%)', 'F1 @2cm (%)', 'F1 @5cm (%)'].some(column => Number.isFinite(parseFloat(row[column])));
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

function setMetricExperiment(experiment) {
  if (!Object.hasOwn(metricMethods, experiment)) return;
  metricExperiment = experiment;
  for (const [id, value] of [['metricsBaseline', 'baseline'], ['metricsAblation', 'scaffold_anchor']]) {
    $(id).setAttribute('aria-pressed', String(value === experiment));
  }
  renderMetrics();
}

function renderMetrics() {
  const method = $('metricMethod').value || 'all';
  const threshold = $('geometryThreshold').value === '5' ? 5 : 2;
  const filtered = cachedMetrics.filter(row => metricMethods[metricExperiment].includes(row.Metode))
    .map(row => ({...row, Metode: row.Metode.replace('_scaffold_anchor', '')}))
    .filter(row => method === 'all' || row.Metode === method);
  const modelGroups = new Map();
  for (const row of filtered) {
    const key = JSON.stringify([row.Kelas, row['Model ID']]);
    if (!modelGroups.has(key)) modelGroups.set(key, []);
    modelGroups.get(key).push(row);
  }
  const display = [...modelGroups.values()].flatMap((rows, groupIndex) =>
    rows.sort((a, b) => ['3DGS', 'SuGaR'].indexOf(a.Metode) - ['3DGS', 'SuGaR'].indexOf(b.Metode))
      .map((row, rowIndex) => ({...row, shadedGroup: groupIndex % 2 === 0, groupStart: rowIndex === 0})));

  const precision = `Precision @${threshold}cm (%)`, recall = `Recall @${threshold}cm (%)`, f1 = `F1 @${threshold}cm (%)`;
  const columns = ['Timestamp', 'Kelas', 'Model ID', 'Metode', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Chamfer Dist (cm)',
    precision, recall, f1, 'Peak VRAM (GB)', 'Total Waktu (Menit)'];
  $('metricsBody').innerHTML = display.map(row => `<tr class="${row.shadedGroup ? 'metric-group-tinted' : 'metric-group-plain'}${row.groupStart ? ' metric-group-start' : ''}">` + columns.map((column, i) => `<td>${column === 'Model ID' ? modelIdentityHtml(row.Kelas || '', row[column] || '', cachedStatus[row.Kelas]?.[row[column]]?.display_name || row['Nama Model'] || defaultModelName(row[column] || '')) : esc(i < 4 ? row[column] || '-' : fmtN(row[column]))}</td>`).join('') + '</tr>').join('') || '<tr><td colspan="13" class="empty-msg"><strong>Belum ada hasil evaluasi</strong>Jalankan evaluasi visual; metrik geometri memerlukan ground truth LiDAR dan kalibrasi skala.</td></tr>';
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
  $('metricScope').textContent = `${metricExperiment === 'baseline' ? 'Baseline' : 'Studi ablasi'} · Rata-rata hasil terakhir per model/metode · ${method === 'all' ? '3DGS + SuGaR' : method}. ${metricExperiment === 'baseline' ? 'Geometri memerlukan LiDAR.' : 'Waktu training belum termasuk tahap Scaffold. Evaluasi geometri ablasi belum tersedia melalui dashboard.'}`;
  for (const [id, label] of [['precisionHeader', 'Precision'], ['recallHeader', 'Recall'], ['f1Header', 'F1']]) {
    $(id).textContent = `${label} @${threshold}cm (%)`;
  }
}
function exportExcel() {
  if (isFilePreview) { showToast(previewMessage); return; }
  window.open('/api/metrics.xlsx', '_blank');
}
function showToast(message) {
  $('toast').textContent = message;
  $('toast').classList.add('show');
  setTimeout(() => $('toast').classList.remove('show'), 6500);
}

async function loadPreview(cls, mdl) {
  if (isFilePreview) return;
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
