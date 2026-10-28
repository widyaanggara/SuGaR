const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function page() {
  const html = fs.readFileSync('web_dashboard/index.html', 'utf8');
  const nodes = new Map();
  const events = {};
  const calls = [];
  function element() {
    return {textContent: '', innerHTML: '', value: '', checked: true, children: [], disabled: false,
      attributes: {}, hidden: false, tabIndex: 0, focused: false,
      setAttribute(name, value) {this.attributes[name] = value;},
      removeAttribute(name) {delete this.attributes[name];}, focus() {this.focused = true;},
      open: false, showModal() {this.open = true;}, close() {this.open = false;},
      classList: {add() {}, remove() {}, toggle() {}}, dataset: {},
      lastElementChild: {children: [], appendChild(child) {this.children.push(child);}},
      appendChild(child) {this.children.push(child);}};
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], element());
  nodes.get('gpuId').value = '2';
  nodes.get('meshTriangles').value = '250000';
  nodes.get('refinementTime').value = 'short';
  nodes.get('gaussiansPerTriangle').value = '1';
  nodes.get('regularization').value = 'dn_consistency';
  nodes.get('evaluationMethod').value = 'SuGaR';
  nodes.get('colmapMaxImageSize').value = '3200';
  nodes.get('imageDownscale').value = '2';
  nodes.get('imageMaxSize').value = '1280';
  nodes.get('colmapDevice').value = 'auto';
  nodes.get('colmapMatcher').value = 'exhaustive';
  nodes.get('colmapMaxFeatures').value = '8192';
  nodes.get('colmapSequentialOverlap').value = '10';
  nodes.get('settingsPreset').value = 'custom';
  nodes.get('sourceWidth').value = '3840';
  nodes.get('sourceHeight').value = '2160';
  nodes.get('metricMethod').value = 'all';
  nodes.get('geometryThreshold').value = '2';
  const storage = new Map();
  const catalog = {classes: Object.fromEntries(['Dewa','Dewi','Mitologi','Penabuh','Pengapit','Punakawan','Raksasa','Wanara'].map(cls => [cls, ['model_01','model_02']])), total_models: 18};
  catalog.classes.TanksAndTemples = ['tanks_and_temples_1','tanks_and_temples_2'];
  const status = Object.fromEntries(Object.entries(catalog.classes).map(([cls, models]) => [cls,
    Object.fromEntries(models.map(mdl => [mdl, {images_count: 0, overall_status: 'No Data', input_path: `datasets/${cls}/${mdl}/input/`, output_path: `outputs/{3dgs,sugar}/${cls}/${mdl}/`}]))]));
  const state = {jobs: [], counts: {}, paused: false, is_running: false, status: 'IDLE', elapsed_seconds: 0, logs: []};
  let jobLogs = {job: {class_name: 'Dewa', model_id: 'model_01', action: 'full_pipeline', status: 'RUNNING'}, logs: ['[COLMAP] Matching 1/10']};
  let failLogs = false;
  let metrics = [];
  const context = vm.createContext({document: {getElementById: id => nodes.get(id) || null,
      createElement: element, querySelectorAll: () => [], addEventListener() {}},
    window: {addEventListener: (name, callback) => events[name] = callback, open() {},
      localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)}},
    navigator: {clipboard: {writeText: async () => {}}}, console,
    setTimeout() {}, setInterval() {},
    fetch: async (url, options) => {
      calls.push([url, options && options.body ? JSON.parse(options.body) : null]);
      if (url.startsWith('/api/jobs/')) return {ok: !failLogs, json: async () => failLogs ? {error: 'Log belum ditemukan'} : jobLogs};
      const body = url === '/api/colmap' ? {available: true, cuda: false, message: 'Mode otomatis memakai CPU.'} :
        url === '/api/presets' ? JSON.parse(fs.readFileSync('web_dashboard/research_presets.json', 'utf8')) :
        url === '/api/catalog' ? catalog : url === '/api/status' ? status : url === '/api/metrics' ? metrics :
        url === '/api/queue' ? state : url.startsWith('/api/previews/') ? {images: []} : {added: ['id'], skipped_empty: [], duplicates: []};
      return {ok: true, json: async () => body};
    }});
  vm.runInContext(fs.readFileSync('web_dashboard/dashboard.js', 'utf8'), context);
  return {nodes, events, calls, context, storage, setMetrics(value) {metrics = value;}, setQueue(value) {Object.assign(state, value);},
    setJobLogs(value) {jobLogs = value;}, failLogs() {failLogs = true;}};
}

test('dashboard initialises all 18 models and 9 dataset groups using actual HTML controls', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  assert.equal(app.nodes.get('kpiTotal').textContent, 18);
  assert.equal(app.nodes.get('kpiCompleted').textContent, '0 / 18');
  assert.equal(app.nodes.get('modelGrid').children.length, 9);
  assert.equal(app.nodes.get('filterBar').children.length, 10);
  assert.match(app.nodes.get('modelGrid').children.at(-1).innerHTML, /tanks_and_temples_2/);
  assert.match(app.nodes.get('modelGrid').children.at(-1).innerHTML, /input\//);
  assert.match(app.nodes.get('queueSummary').textContent, /0 menunggu/);
  const firstCard = app.nodes.get('modelGrid').children[0];
  await vm.runInContext('refreshStatus(false)', app.context);
  assert.equal(app.nodes.get('modelGrid').children[0], firstCard,
    'Polling unchanged status must preserve open folder details and clickable model controls');
});

test('settings tabs support keyboard navigation without losing hidden settings or sending jobs', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  assert.equal(app.nodes.get('pane-resolution').hidden, false);
  assert.equal(app.nodes.get('pane-colmap').hidden, true);
  const event = key => ({key, preventDefault() {this.prevented = true;}});
  const right = event('ArrowRight');
  app.nodes.get('tab-resolution').onkeydown(right);
  assert.equal(right.prevented, true);
  assert.equal(app.nodes.get('pane-resolution').hidden, true);
  assert.equal(app.nodes.get('pane-colmap').hidden, false);
  assert.equal(app.nodes.get('tab-colmap').attributes['aria-selected'], 'true');
  assert.equal(app.nodes.get('tab-colmap').focused, true);
  assert.equal(app.nodes.get('tab-resolution').tabIndex, -1);
  app.nodes.get('colmapMaxFeatures').value = '4096';
  app.nodes.get('tab-colmap').onkeydown(event('End'));
  assert.equal(app.nodes.get('pane-lidar').hidden, false);
  app.nodes.get('tab-lidar').onkeydown(event('ArrowRight'));
  assert.equal(app.nodes.get('pane-resolution').hidden, false);
  assert.equal(app.nodes.get('colmapMaxFeatures').value, '4096');
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  await vm.runInContext("triggerRun('colmap', 'Dewa', 'model_01')", app.context);
  assert.equal(app.calls.find(([url]) => url === '/api/run')[1].colmap_max_features, 4096);
});

test('visual evaluation enqueues both methods while other work can be active', async () => {
  const app = page();
  await vm.runInContext("triggerRun('evaluate_visual', 'Dewa', 'model_01')", app.context);
  const requests = app.calls.filter(([url]) => url === '/api/run');
  assert.equal(requests.length, 2);
  assert.deepEqual(requests.map(([, body]) => body.evaluation_method), ['3DGS', 'SuGaR']);
  assert.equal(requests[0][1].gpu, 2);
  assert.equal(requests[0][1].resume, true);
  assert.equal(requests[0][1].colmap_max_image_size, 3200);
  assert.equal(requests[0][1].image_downscale, 2);
  assert.equal(requests[0][1].image_max_size, 1280);
  assert.equal(requests[0][1].colmap_device, 'auto');
  assert.equal(requests[0][1].colmap_max_features, 8192);
});

test('upstream presets reach the queued pipeline without overriding data and COLMAP settings', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  for (const [preset, refinement, budget, gaussians] of [
    ['official_high', 'long', 1000000, 1],
    ['official_low', 'long', 200000, 6],
    ['official_short', 'short', 200000, 6]]) {
    vm.runInContext(`applyPreset('${preset}')`, app.context);
    assert.match(app.nodes.get('resolutionPreview').textContent, /1280 × 720/);
    await vm.runInContext("triggerRun('full_pipeline', 'Punakawan', 'model_01')", app.context);
    const body = app.calls.filter(([url]) => url === '/api/run').at(-1)[1];
    assert.equal(body.colmap_max_image_size, 3200);
    assert.equal(body.image_max_size, 1280);
    assert.equal(body.colmap_max_features, 8192);
    assert.equal(body.refinement_time, refinement);
    assert.equal(body.vertices, budget);
    assert.equal(body.gaussians_per_triangle, gaussians);
    assert.equal(body.regularization, 'dn_consistency');
    assert.equal(body.colmap_device, 'auto');
    assert.equal(body.colmap_matcher, 'exhaustive');
  }
});

test('custom settings survive refresh and remain separate from already submitted jobs', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  vm.runInContext("applyPreset('official_high')", app.context);
  await vm.runInContext("triggerRun('colmap', 'Dewa', 'model_01')", app.context);
  app.nodes.get('colmapMatcher').value = 'sequential';
  app.nodes.get('colmapMatcher').onchange();
  assert.equal(app.nodes.get('colmapSequentialOverlap').disabled, false);
  assert.equal(app.nodes.get('settingsPreset').value, 'official_high');
  app.nodes.get('imageDownscale').value = '4';
  app.nodes.get('imageDownscale').onchange();
  assert.match(app.nodes.get('resolutionPreview').textContent, /960 × 540/);
  const saved = JSON.parse(app.storage.get('sugar-experiment-settings-v1'));
  assert.equal(saved.imageDownscale, '4');
  app.nodes.get('imageDownscale').value = '1';
  vm.runInContext('initializeSettings()', app.context);
  assert.equal(app.nodes.get('imageDownscale').value, '4');
  assert.equal(app.calls.find(([url]) => url === '/api/run')[1].image_downscale, 2);
  assert.match(app.nodes.get('colmapCapability').textContent, /otomatis memakai CPU/);
});

test('legacy pilot settings are preserved and clearly marked custom instead of claiming upstream provenance', async () => {
  const app = page();
  app.storage.set('sugar-experiment-settings-v1', JSON.stringify({settingsPreset: 'pilot',
    meshTriangles: '250000', refinementTime: 'short', imageMaxSize: '1600'}));
  await app.events.DOMContentLoaded();
  assert.equal(app.nodes.get('settingsPreset').value, 'custom');
  assert.equal(app.nodes.get('meshTriangles').value, '250000');
  assert.match(app.nodes.get('presetDescription').textContent, /tidak diklaim sebagai preset resmi/);
});

test('geometry evaluation rejects absent scale before submitting a job', async () => {
  const app = page();
  await vm.runInContext("triggerRun('evaluate', 'Dewa', 'model_01')", app.context);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  assert.match(app.nodes.get('toast').textContent, /kalibrasi/);
});

test('all seven metric cards show real merged results with method and geometry-threshold filters', async () => {
  const app = page();
  const base = {Kelas: 'Dewa', 'Model ID': 'model_01', Status: 'COMPLETED'};
  app.setMetrics([
    {...base, Metode: '3DGS', 'PSNR (dB)': '20', SSIM: '.8', LPIPS: '.2'},
    {...base, Metode: 'SuGaR', 'PSNR (dB)': '30', SSIM: '.9', LPIPS: '.1'},
    {...base, Metode: 'SuGaR', 'Chamfer Dist (cm)': '1.2', 'PSNR (dB)': '-',
      'Precision @2cm (%)': '80', 'Recall @2cm (%)': '50', 'F1 @2cm (%)': '61.5385',
      'Precision @5cm (%)': '100', 'Recall @5cm (%)': '80', 'F1 @5cm (%)': '88.8889'},
    {...base, Metode: '3DGS', 'PSNR (dB)': '999', Status: 'FAILED'}]);
  await vm.runInContext('refreshMetrics()', app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '25.00 dB');
  assert.equal(app.nodes.get('kpiSsim').textContent, '0.8500');
  assert.equal(app.nodes.get('kpiLpips').textContent, '0.1500');
  assert.equal(app.nodes.get('kpiChamfer').textContent, '1.20 cm');
  assert.equal(app.nodes.get('kpiPrecision').textContent, '80.00%');
  assert.equal(app.nodes.get('kpiRecall').textContent, '50.00%');
  assert.equal(app.nodes.get('kpiF1').textContent, '61.54%');
  assert.match(app.nodes.get('precisionContext').textContent, /1 hasil.*2 cm/);
  app.nodes.get('geometryThreshold').value = '5';
  vm.runInContext('renderMetrics()', app.context);
  assert.equal(app.nodes.get('kpiF1').textContent, '88.89%');
  assert.equal(app.nodes.get('precisionHeader').textContent, 'Precision @5cm (%)');
  app.nodes.get('metricMethod').value = '3DGS';
  vm.runInContext('renderMetrics()', app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '20.00 dB');
  assert.equal(app.nodes.get('kpiF1').textContent, '—');
  assert.doesNotMatch(app.nodes.get('metricsBody').innerHTML, /SuGaR/);
});

test('missing geometry remains unavailable while measured zero scores stay zero', async () => {
  const app = page();
  await vm.runInContext('refreshMetrics()', app.context);
  for (const id of ['kpiPsnr','kpiSsim','kpiLpips','kpiChamfer','kpiPrecision','kpiRecall','kpiF1']) {
    assert.equal(app.nodes.get(id).textContent, '—');
  }
  app.setMetrics([{Kelas: 'Dewa', 'Model ID': 'model_01', Metode: 'SuGaR', Status: 'COMPLETED',
    'PSNR (dB)': 'NaN', SSIM: '0', LPIPS: '0', 'Precision @2cm (%)': '0', 'Recall @2cm (%)': '0', 'F1 @2cm (%)': '0'}]);
  await vm.runInContext('refreshMetrics()', app.context);
  assert.equal(app.nodes.get('kpiF1').textContent, '0.00%');
  assert.equal(app.nodes.get('kpiSsim').textContent, '0.0000');
  assert.equal(app.nodes.get('kpiLpips').textContent, '0.0000');
  assert.equal(app.nodes.get('kpiPsnr').textContent, '—');
  assert.equal(app.nodes.get('kpiChamfer').textContent, '—');
});

test('Log button opens a visible dialog and selected logs keep updating during polling', async () => {
  const app = page();
  await vm.runInContext("showJobLogs('test-id')", app.context);
  assert.equal(app.nodes.get('jobLogDialog').open, true);
  assert.match(app.nodes.get('jobLogTitle').textContent, /Dewa\/model_01/);
  assert.match(app.nodes.get('jobLogContent').textContent, /Matching 1\/10/);
  app.setJobLogs({job: {class_name: 'Dewa', model_id: 'model_01', action: 'full_pipeline', status: 'RUNNING'}, logs: ['[COLMAP] Matching 2/10']});
  await vm.runInContext('pollLogs()', app.context);
  assert.match(app.nodes.get('jobLogContent').textContent, /Matching 2\/10/);
  vm.runInContext('closeJobLogs()', app.context);
  assert.equal(app.nodes.get('jobLogDialog').open, false);
  const before = app.calls.filter(([url]) => url.startsWith('/api/jobs/')).length;
  await vm.runInContext('pollLogs()', app.context);
  assert.equal(app.calls.filter(([url]) => url.startsWith('/api/jobs/')).length, before);
});

test('log errors are visible in the dialog instead of leaving an apparently dead button', async () => {
  const app = page();
  app.failLogs();
  await vm.runInContext("showJobLogs('missing-id')", app.context);
  assert.equal(app.nodes.get('jobLogDialog').open, true);
  assert.equal(app.nodes.get('jobLogStatus').textContent, 'Gagal memuat log');
  assert.equal(app.nodes.get('jobLogContent').textContent, 'Log belum ditemukan');
});

test('polling keeps the actual Log button stable so clicks are not lost while a job runs', async () => {
  const app = page();
  app.setQueue({is_running: true, status: 'RUNNING', jobs: [{id: 'live', class_name: 'Punakawan',
    model_id: 'model_01', action: 'full_pipeline', status: 'RUNNING', started_at: 1}]});
  await vm.runInContext('pollLogs()', app.context);
  const row = app.nodes.get('queueBody').children[0];
  const button = row.lastElementChild.children[0];
  assert.equal(button.textContent, 'Log');
  await vm.runInContext('pollLogs()', app.context);
  assert.equal(app.nodes.get('queueBody').children[0], row);
  assert.equal(row.lastElementChild.children[0], button);
  await button.onclick();
  assert.equal(app.nodes.get('jobLogDialog').open, true);
});
