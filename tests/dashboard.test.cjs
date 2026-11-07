const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function page(localPreview = false) {
  const html = fs.readFileSync('web_dashboard/index.html', 'utf8');
  const nodes = new Map();
  const events = {};
  const calls = [];
  function element() {
    return {textContent: '', _innerHTML: '', value: '', checked: true, children: [], disabled: false,
      get innerHTML() {return this._innerHTML;},
      set innerHTML(value) {this._innerHTML = value; this.children = [];},
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
  let failModelName = false;
  let metrics = [];
  const context = vm.createContext({document: {getElementById: id => nodes.get(id) || null,
      createElement: element, querySelectorAll: () => [], addEventListener() {}},
    window: {addEventListener: (name, callback) => events[name] = callback, open() {},
      location: {protocol: localPreview ? 'file:' : 'http:'},
      localStorage: {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value)}},
    navigator: {clipboard: {writeText: async () => {}}}, console,
    setTimeout() {}, setInterval() {},
    fetch: async (url, options) => {
      calls.push([url, options && options.body ? JSON.parse(options.body) : null]);
      if (url.startsWith('/api/jobs/')) return {ok: !failLogs, json: async () => failLogs ? {error: 'Log belum ditemukan'} : jobLogs};
      if (url === '/api/geometry-scales') {
        const scales = {};
        for (const entry of JSON.parse(options.body).scales) {
          status[entry.class_name][entry.model_id].geometry_scale_factor = entry.scale_factor;
          if (entry.scale_factor != null) (scales[entry.class_name] ||= {})[entry.model_id] = entry.scale_factor;
        }
        return {ok: true, json: async () => ({scales})};
      }
      if (url === '/api/geometry-settings') {
        const settings = {};
        for (const entry of JSON.parse(options.body).settings) {
          const value = entry.clear ? null : {lidar_unit: entry.lidar_unit,
            alignment_mode: entry.alignment_mode, ...(entry.scale_factor == null ? {} : {scale_factor: entry.scale_factor})};
          status[entry.class_name][entry.model_id].geometry_settings = value;
          if (value) (settings[entry.class_name] ||= {})[entry.model_id] = value;
        }
        return {ok: true, json: async () => ({settings})};
      }
      if (url === '/api/model-name') {
        if (failModelName) return {ok: false, json: async () => ({error: 'Perubahan tidak dapat disimpan.'})};
        const body = JSON.parse(options.body);
        const customName = body.display_name.trim();
        const result = {class_name: body.class_name, model_id: body.model_id, custom_name: customName,
          display_name: customName || 'Model ' + body.model_id.slice(6)};
        Object.assign(status[body.class_name][body.model_id], result);
        return {ok: true, json: async () => result};
      }
      const body = url === '/api/colmap' ? {available: true, cuda: false, message: 'Mode otomatis memakai CPU.'} :
        url === '/api/presets' ? JSON.parse(fs.readFileSync('web_dashboard/research_presets.json', 'utf8')) :
        url === '/api/catalog' ? catalog : url === '/api/status' ? status : url === '/api/metrics' ? metrics :
        url === '/api/queue' ? state : url.startsWith('/api/previews/') ? {images: []} : {added: ['id'], skipped_empty: [], duplicates: []};
      return {ok: true, json: async () => body};
    }});
  vm.runInContext(fs.readFileSync('web_dashboard/dashboard.js', 'utf8'), context);
  return {nodes, events, calls, context, storage, setMetrics(value) {metrics = value;}, setQueue(value) {Object.assign(state, value);},
    setJobLogs(value) {jobLogs = value;}, failLogs() {failLogs = true;}, failModelName() {failModelName = true;},
    setModel(cls, mdl, values) {Object.assign(status[cls][mdl], values);}};
}

test('queue popup styles are bundled in the HTML so missing CSS routes cannot strip its layout', () => {
  const html = fs.readFileSync('web_dashboard/index.html', 'utf8');
  const styles = [...html.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/g)].map(match => match[1]).join('\n');
  assert.doesNotMatch(html, /<link\b[^>]*href="[^\"]*queue\.css"/);
  for (const selector of ['.queue-dialog{', '.queue-dialog-form{', '.queue-dialog-head{',
    '.queue-target-fields{', '.process-options{', '.process-option{', '.queue-dialog-footer{']) {
    assert.ok(styles.includes(selector), `Core popup layout must ship with the HTML: ${selector}`);
  }
  assert.match(styles, /\.queue-dialog-content\{[^}]*overflow-y:auto/);
  assert.match(styles, /@media\(max-width:600px\)/);
});

test('opening HTML directly supports settings tabs and resolution previews without backend requests', async () => {
  const app = page(true);
  await app.events.DOMContentLoaded();
  assert.equal(app.nodes.get('previewNotice').hidden, false);
  vm.runInContext("openQueueDialog('', '', 'evaluate')", app.context);
  assert.equal(app.nodes.get('queueDialog').open, true);
  assert.equal(app.nodes.get('queueDialogSubmit').disabled, true);
  assert.equal(app.nodes.get('settingsPreset').disabled, true);
  assert.match(app.nodes.get('resolutionPreview').textContent, /1280 × 720/);
  vm.runInContext("switchSettingsTab('colmap')", app.context);
  assert.equal(app.nodes.get('pane-colmap').hidden, false);
  assert.equal(app.nodes.get('pane-resolution').hidden, true);
  app.nodes.get('imageDownscale').value = '4';
  app.nodes.get('imageDownscale').onchange();
  assert.match(app.nodes.get('resolutionPreview').textContent, /960 × 540/);
  assert.match(app.nodes.get('summaryTraining').textContent, /4×/);
  await vm.runInContext("triggerRun('full_pipeline', 'Dewa', 'model_01')", app.context);
  assert.match(app.nodes.get('toast').textContent, /Pratinjau desain lokal/);
  assert.equal(app.calls.length, 0);
  const html = fs.readFileSync('web_dashboard/index.html', 'utf8');
  assert.match(html, /src="\.\/dashboard\.js"/);
});

test('editing a name persists through refresh and labels results without changing queued model IDs', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  vm.runInContext("editModelName('Dewa', 'model_01')", app.context);
  assert.equal(app.nodes.get('modelNameDialog').open, true);
  assert.equal(app.nodes.get('modelNameTarget').textContent, 'Dewa / model_01');
  app.nodes.get('modelNameInput').value = '  Ganesha  ';
  await vm.runInContext('saveModelName()', app.context);
  const body = app.calls.find(([url]) => url === '/api/model-name')[1];
  assert.deepEqual(body, {class_name: 'Dewa', model_id: 'model_01', display_name: 'Ganesha'});
  assert.equal(app.nodes.get('modelNameDialog').open, false);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  await vm.runInContext('refreshStatus(false)', app.context);
  assert.match(app.nodes.get('modelGrid').children[0].innerHTML, /Ganesha/);
  assert.match(app.nodes.get('modelGrid').children[0].innerHTML, /datasets\/Dewa\/model_01\/input\//);
  app.setQueue({jobs: [{id: 'named-job', class_name: 'Dewa', model_id: 'model_01', action: 'colmap', status: 'QUEUED'}]});
  await vm.runInContext('pollLogs()', app.context);
  assert.match(app.nodes.get('queueBody').children[0].innerHTML, /Ganesha/);
  app.setMetrics([{Kelas: 'Dewa', 'Model ID': 'model_01', Metode: 'SuGaR', Status: 'COMPLETED', 'PSNR (dB)': '24.5'}]);
  await vm.runInContext('refreshMetrics()', app.context);
  assert.match(app.nodes.get('metricsBody').innerHTML, /Ganesha/);
  assert.match(app.nodes.get('metricsBody').innerHTML, /Dewa\/model_01/);
  await vm.runInContext("triggerRun('colmap', 'Dewa', 'model_01')", app.context);
  assert.equal(app.calls.find(([url]) => url === '/api/run')[1].model_id, 'model_01');
});

test('names are escaped as text and can be reset or cancelled', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  vm.runInContext("editModelName('Dewa', 'model_01')", app.context);
  app.nodes.get('modelNameInput').value = '<img src=x onerror=alert(1)>';
  await vm.runInContext('saveModelName()', app.context);
  assert.match(app.nodes.get('modelGrid').children[0].innerHTML, /&lt;img src=x/);
  assert.doesNotMatch(app.nodes.get('modelGrid').children[0].innerHTML, /<img src=x/);
  vm.runInContext("editModelName('Dewa', 'model_01')", app.context);
  app.nodes.get('modelNameInput').value = '';
  await vm.runInContext('saveModelName()', app.context);
  assert.match(app.nodes.get('modelGrid').children[0].innerHTML, /Model 01/);
  const before = app.calls.filter(([url]) => url === '/api/model-name').length;
  vm.runInContext("editModelName('Dewa', 'model_01'); closeModelName()", app.context);
  assert.equal(app.nodes.get('modelNameDialog').open, false);
  assert.equal(app.calls.filter(([url]) => url === '/api/model-name').length, before);
});

test('failed name saves keep the dialog open with an error and do not enqueue work', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  app.failModelName();
  vm.runInContext("editModelName('Dewa', 'model_01')", app.context);
  app.nodes.get('modelNameInput').value = 'Ganesha';
  await vm.runInContext('saveModelName()', app.context);
  assert.equal(app.nodes.get('modelNameDialog').open, true);
  assert.equal(app.nodes.get('modelNameError').textContent, 'Perubahan tidak dapat disimpan.');
  assert.equal(app.nodes.get('modelNameSave').disabled, false);
  assert.equal(app.nodes.get('modelNameInput').disabled, false);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
});

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

test('settings summary and submitted jobs read the same controls; photo reference stays preview-only', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  for (const [id, value] of Object.entries({gpuId: '0', colmapDevice: 'cpu', colmapMaxImageSize: '2000',
    imageDownscale: '1', imageMaxSize: '1600', refinementTime: 'long', sourceWidth: '1920', sourceHeight: '1080'})) {
    app.nodes.get(id).value = value;
    app.nodes.get(id).onchange();
  }
  assert.equal(app.nodes.get('summaryGpu').textContent, 'GPU 0');
  assert.equal(app.nodes.get('summaryColmap').textContent, 'CPU / maks. 2000 px');
  assert.equal(app.nodes.get('summaryTraining').textContent, '1× / maks. 1600 px');
  assert.equal(app.nodes.get('summaryRefinement').textContent, '15.000 iterasi');
  assert.match(app.nodes.get('resolutionPreview').textContent, /1600 × 900/);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  await vm.runInContext("triggerRun('full_pipeline', 'Dewa', 'model_01')", app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.gpu, 0);
  assert.equal(body.colmap_device, 'cpu');
  assert.equal(body.colmap_max_image_size, 2000);
  assert.equal(body.image_downscale, 1);
  assert.equal(body.image_max_size, 1600);
  assert.equal(body.refinement_time, 'long');
  assert.equal(Object.hasOwn(body, 'sourceWidth'), false);
  assert.equal(Object.hasOwn(body, 'sourceHeight'), false);
  app.nodes.get('imageMaxSize').value = '0';
  app.nodes.get('imageMaxSize').onchange();
  assert.match(app.nodes.get('summaryTraining').textContent, /tanpa batas tambahan/);
  assert.equal(body.image_max_size, 1600, 'Changing a control must not change an already submitted job');
});

test('vertical settings navigation supports up and down arrows', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  assert.equal(app.nodes.get('settingsTabList').attributes['aria-orientation'], 'vertical');
  let prevented = false;
  app.nodes.get('tab-resolution').onkeydown({key: 'ArrowDown', preventDefault() {prevented = true;}});
  assert.equal(prevented, true);
  assert.equal(app.nodes.get('pane-colmap').hidden, false);
  assert.equal(app.nodes.get('tab-colmap').focused, true);
  app.nodes.get('tab-colmap').onkeydown({key: 'ArrowUp', preventDefault() {}});
  assert.equal(app.nodes.get('pane-resolution').hidden, false);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
});

test('model cards open a process chooser instead of exposing many training buttons', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {images_count: 20});
  await app.events.DOMContentLoaded();
  const card = app.nodes.get('modelGrid').children[0].innerHTML;
  assert.match(card, /Antrekan model ini/);
  assert.doesNotMatch(card, /2\. COLMAP|3\. Latih 3DGS|4\. Latih SuGaR/);
  assert.equal(app.nodes.has('btnRunAllVisual'), false);
  assert.equal(app.nodes.has('btnRunAllGeometry'), false);
  assert.equal(app.nodes.has('btnRunAll'), false);
  assert.equal(app.nodes.has('btnQueueModels'), true);
  vm.runInContext("openQueueDialog('Dewa', 'model_01')", app.context);
  assert.equal(app.nodes.get('queueDialog').open, true);
  assert.equal(app.nodes.get('queueMethodOptions').hidden, true);
  assert.match(app.nodes.get('queuePlanSummary').textContent, /1 pekerjaan/);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  await vm.runInContext('submitQueueDialog()', app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.action, 'full_pipeline');
  assert.equal(body.class_name, 'Dewa');
  assert.equal(body.model_id, 'model_01');
  assert.equal(app.nodes.get('queueDialog').open, false);
});

test('bulk visual evaluation sends both methods without requesting training', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {'3dgs_30k': true, '3dgs_7k': true, sugar_obj: true});
  app.nodes.get('evaluationMethod').value = 'both';
  await app.events.DOMContentLoaded();
  vm.runInContext("openQueueDialog('', '', 'evaluate_visual')", app.context);
  assert.match(app.nodes.get('queuePlanSummary').textContent, /2 pekerjaan/);
  assert.equal(app.nodes.get('queueGeometryOptions').hidden, true);
  await vm.runInContext('submitQueueDialog()', app.context);
  const requests = app.calls.filter(([url]) => url === '/api/run');
  assert.equal(requests.length, 1);
  assert.equal(requests[0][1].action, 'evaluate_visual');
  assert.equal(requests[0][1].evaluation_method, 'both');
  assert.equal(requests[0][1].class_name, '');
});

test('one bulk job per ready model continues from imported COLMAP through visual evaluation', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {images_count: 30, colmap_done: true});
  app.setModel('Dewa', 'model_02', {images_count: 30, colmap_done: false});
  await app.events.DOMContentLoaded();
  vm.runInContext('openQueueDialog()', app.context);
  assert.equal(app.nodes.get('queuePipelineOptions').hidden, false);
  app.nodes.get('queuePipelineStart').value = 'existing_colmap';
  vm.runInContext('updateQueuePlan()', app.context);
  assert.match(app.nodes.get('queuePlanSummary').textContent, /1 pekerjaan/);
  assert.match(app.nodes.get('queueSkippedList').innerHTML, /COLMAP belum lengkap/);
  assert.match(app.nodes.get('queuePipelineHelp').textContent, /COLMAP tidak dijalankan ulang/);
  await vm.runInContext('submitQueueDialog()', app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.action, 'full_pipeline');
  assert.equal(body.pipeline_start, 'existing_colmap');
  assert.equal(body.class_name, '');
  app.nodes.get('queuePipelineStart').value = 'auto';
  assert.equal(body.pipeline_start, 'existing_colmap');
});

test('automatic bulk pipeline includes mixed starting points without a history filter', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {images_count: 30, colmap_done: true});
  app.setModel('Dewa', 'model_02', {images_count: 30, colmap_done: false});
  await app.events.DOMContentLoaded();
  vm.runInContext('openQueueDialog()', app.context);
  assert.equal(app.nodes.get('queuePipelineStart').value, 'auto');
  assert.match(app.nodes.get('queuePlanSummary').textContent, /2 pekerjaan/);
  assert.match(app.nodes.get('queuePipelineHelp').textContent, /COLMAP lengkap, termasuk hasil teman, langsung dipakai/);
  await vm.runInContext('submitQueueDialog()', app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.pipeline_start, 'auto');
  assert.equal(body.resume, true);
  assert.equal(body.class_name, '');
});

test('COLMAP start option is limited to full pipelines and never leaks into evaluation jobs', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {'3dgs_30k': true});
  await app.events.DOMContentLoaded();
  vm.runInContext("openQueueDialog('Dewa', 'model_01')", app.context);
  app.nodes.get('queuePipelineStart').value = 'existing_colmap';
  vm.runInContext("chooseQueueAction('evaluate_visual')", app.context);
  assert.equal(app.nodes.get('queuePipelineOptions').hidden, true);
  app.nodes.get('queueMethod').value = '3DGS';
  await vm.runInContext('submitQueueDialog()', app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.action, 'evaluate_visual');
  assert.equal(body.pipeline_start, 'auto');
});

test('retrying an interrupted model uses its server ID to request priority recovery', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  app.setQueue({jobs: [{id: 'imported', class_name: 'Dewa', model_id: 'model_01',
    action: 'full_pipeline', status: 'INTERRUPTED', pipeline_start: 'existing_colmap', resume: true}]});
  await vm.runInContext('pollLogs()', app.context);
  const row = app.nodes.get('queueBody').children[0];
  assert.match(row.innerHTML, /Mulai dari hasil COLMAP/);
  await row.lastElementChild.children[1].onclick();
  const body = app.calls.find(([url]) => url === '/api/queue/retry')[1];
  assert.equal(body.job_id, 'imported');
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  assert.match(app.nodes.get('toast').textContent, /diprioritaskan/);
});

test('paused queue displays recovery first among waiting jobs while active work stays above it', async () => {
  const app = page();
  await app.events.DOMContentLoaded();
  const ordinary = {id: 'ordinary', class_name: 'Dewa', model_id: 'model_01', action: 'full_pipeline', status: 'QUEUED', created_at: 1};
  const recovery = {id: 'recovery', class_name: 'Dewa', model_id: 'model_02', action: 'full_pipeline', status: 'QUEUED', created_at: 9, recovery: true, recovery_requested_at: 9};
  app.setQueue({paused: true, status: 'PAUSED', jobs: [ordinary, recovery]});
  await vm.runInContext('pollLogs()', app.context);
  assert.match(app.nodes.get('queueBody').children[0].innerHTML, /Dewa\/model_02/);
  assert.match(app.nodes.get('queueBody').children[0].innerHTML, /Prioritas pemulihan/);
  assert.match(app.nodes.get('queueBody').children[1].innerHTML, /Dewa\/model_01/);
  assert.equal(app.nodes.get('btnPause').textContent, 'Lanjutkan Antrean');
  assert.equal(app.calls.filter(([url]) => url === '/api/queue/resume').length, 0);
  const active = {id: 'active', class_name: 'Dewi', model_id: 'model_01', action: 'full_pipeline', status: 'RUNNING', queue_position: 1};
  app.setQueue({paused: false, is_running: true, jobs: [ordinary, {...recovery, queue_position: 2}, active]});
  await vm.runInContext('pollLogs()', app.context);
  assert.match(app.nodes.get('queueBody').children[0].innerHTML, /Dewi\/model_01/);
  assert.match(app.nodes.get('queueBody').children[1].innerHTML, /Dewa\/model_02/);
});

test('bulk geometry persists each LiDAR unit and auto alignment without asking for scale', async () => {
  const app = page();
  for (const [mdl, scale] of [['model_01', 0.02], ['model_02', 0.07]]) {
    app.setModel('Dewa', mdl, {has_gt_lidar: true, geometry_sources: {'3DGS': true, SuGaR: true}, geometry_scale_factor: scale,
      geometry_settings: {lidar_unit: mdl === 'model_01' ? 'm' : 'mm', alignment_mode: 'automatic'}});
  }
  app.nodes.get('evaluationMethod').value = 'both';
  await app.events.DOMContentLoaded();
  vm.runInContext("openQueueDialog('', '', 'evaluate')", app.context);
  assert.equal(app.nodes.get('geometryCalibrationBody').children.length, 18);
  assert.match(app.nodes.get('queuePlanSummary').textContent, /4 pekerjaan/);
  assert.equal(vm.runInContext("geometryInputs.get('Dewa/model_01').unit.value", app.context), 'm');
  assert.equal(vm.runInContext("geometryInputs.get('Dewa/model_02').unit.value", app.context), 'mm');
  assert.equal(vm.runInContext("geometryInputs.get('Dewi/model_01').unit.value", app.context), '');
  assert.equal(app.nodes.get('geometryScaleHeading').hidden, true);
  app.nodes.get('queueApplyIcp').checked = false;
  await vm.runInContext('submitQueueDialog()', app.context);
  const calibration = app.calls.find(([url]) => url === '/api/geometry-settings');
  const enqueue = app.calls.find(([url]) => url === '/api/run');
  assert.ok(app.calls.indexOf(calibration) < app.calls.indexOf(enqueue));
  assert.equal(calibration[1].settings.find(item => item.class_name === 'Dewa' && item.model_id === 'model_01').lidar_unit, 'm');
  assert.equal(calibration[1].settings.find(item => item.class_name === 'Dewa' && item.model_id === 'model_02').lidar_unit, 'mm');
  assert.equal(calibration[1].settings.find(item => item.class_name === 'Dewa' && item.model_id === 'model_01').scale_factor, undefined);
  assert.equal(enqueue[1].action, 'evaluate');
  assert.equal(enqueue[1].evaluation_method, 'both');
  assert.equal(enqueue[1].apply_icp, true);
  assert.equal(enqueue[1].geometry_settings_required, true);
  assert.equal(Object.hasOwn(enqueue[1], 'scale_factor'), false);
});

test('geometry requires an explicit LiDAR unit but automatic mode never requires a manual scale', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {has_gt_lidar: true, geometry_sources: {'3DGS': true, SuGaR: true}});
  await app.events.DOMContentLoaded();
  vm.runInContext("openQueueDialog('Dewa', 'model_01', 'evaluate')", app.context);
  assert.equal(app.nodes.get('queueGeometryOptions').hidden, false);
  assert.equal(app.nodes.get('queueDialogSubmit').disabled, true);
  await vm.runInContext('submitQueueDialog()', app.context);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  assert.equal(app.calls.filter(([url]) => url === '/api/geometry-scales').length, 0);
  assert.match(app.nodes.get('queueSkippedList').innerHTML, /satuan/);
  vm.runInContext("geometryInputs.get('Dewa/model_01').unit.value = 'm'; updateQueuePlan()", app.context);
  assert.equal(app.nodes.get('queueDialogSubmit').disabled, false);
  app.nodes.get('queueAlignment').value = 'manual';
  vm.runInContext('updateQueuePlan()', app.context);
  assert.equal(app.nodes.get('queueDialogSubmit').disabled, true);
  assert.equal(app.nodes.get('geometryScaleHeading').hidden, false);
  vm.runInContext("geometryInputs.get('Dewa/model_01').scale.value = '0.02'; updateQueuePlan()", app.context);
  assert.equal(app.nodes.get('queueDialogSubmit').disabled, false);
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

test('legacy geometry action opens the new unit and alignment chooser without submitting a job', async () => {
  const app = page();
  await vm.runInContext("triggerRun('evaluate', 'Dewa', 'model_01')", app.context);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  assert.equal(app.nodes.get('queueDialog').open, true);
  assert.equal(app.nodes.get('queueGeometryOptions').hidden, false);
});

test('topbar chooser changes between all targets, a class, and a single model', async () => {
  const app = page();
  app.setModel('Dewa', 'model_01', {images_count: 20});
  await app.events.DOMContentLoaded();
  vm.runInContext('openQueueDialog()', app.context);
  assert.equal(app.nodes.get('queueDialog').open, true);
  assert.equal(app.nodes.get('queueTargetCount').textContent, '18 model dipilih');
  assert.equal(app.nodes.get('queueModel').disabled, true);
  app.nodes.get('queueClass').value = 'Dewa';
  vm.runInContext('changeQueueTarget(true)', app.context);
  assert.equal(app.nodes.get('queueTargetCount').textContent, '2 model dipilih');
  app.nodes.get('queueModel').value = 'model_01';
  vm.runInContext('changeQueueTarget(false)', app.context);
  assert.equal(app.nodes.get('queueTargetCount').textContent, '1 model dipilih');
  assert.equal(app.nodes.get('queue-action-colmap').hidden, false);
  await vm.runInContext('submitQueueDialog()', app.context);
  const body = app.calls.find(([url]) => url === '/api/run')[1];
  assert.equal(body.class_name, 'Dewa');
  assert.equal(body.model_id, 'model_01');
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
  assert.match(app.nodes.get('metricsHead').innerHTML, /Precision @5cm/);
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


test('evaluation tabs show one row per statue and Excel exports all experiments regardless of filters', async () => {
  const app = page();
  const base = {Kelas: 'Dewa', 'Model ID': 'model_01', Status: 'COMPLETED'};
  app.setMetrics([
    {...base, Metode: '3DGS', 'PSNR (dB)': '20', 'Chamfer Dist (cm)': '1'},
    {...base, Metode: 'SuGaR', 'PSNR (dB)': '30'},
    {...base, Metode: '3DGS_scaffold_anchor', 'PSNR (dB)': '40'},
    {...base, Metode: 'SuGaR_scaffold_anchor', 'PSNR (dB)': '50'},
    {...base, Metode: 'SuGaR_scaffold_anchor', 'PSNR (dB)': '999', Status: 'FAILED'}
  ]);
  await vm.runInContext('refreshMetrics()', app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '25.00 dB');
  vm.runInContext("setMetricExperiment('scaffold_anchor')", app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '45.00 dB');
  assert.doesNotMatch(app.nodes.get('metricsBody').innerHTML, /scaffold_anchor/);
  assert.match(app.nodes.get('metricsBody').innerHTML, /3DGS/);
  assert.match(app.nodes.get('metricsBody').innerHTML, /SuGaR/);
  assert.equal((app.nodes.get('metricsBody').innerHTML.match(/<tr /g) || []).length, 1);
  assert.match(app.nodes.get('metricsHead').innerHTML, /colspan="10"/);
  assert.equal(app.nodes.get('metricsAblation').attributes['aria-pressed'], 'true');
  assert.equal(app.nodes.get('kpiChamfer').textContent, '\u2014');
  app.nodes.get('metricMethod').value = '3DGS';
  vm.runInContext('renderMetrics()', app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '40.00 dB');
  let exported;
  app.context.window.open = url => {exported = url;};
  vm.runInContext('exportExcel()', app.context);
  assert.equal(exported, '/api/metrics.xlsx');
  vm.runInContext("setMetricExperiment('baseline')", app.context);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '20.00 dB');
  app.setMetrics([{...base, Metode: '3DGS', 'PSNR (dB)': '20'}]);
  await vm.runInContext('refreshMetrics()', app.context);
  vm.runInContext("setMetricExperiment('scaffold_anchor')", app.context);
  assert.match(app.nodes.get('metricsBody').innerHTML, /Belum ada hasil evaluasi/);
  assert.equal(app.nodes.get('kpiPsnr').textContent, '\u2014');
});
