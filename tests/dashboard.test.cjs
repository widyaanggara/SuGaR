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
      classList: {add() {}, remove() {}, toggle() {}}, dataset: {}, lastElementChild: {appendChild() {}},
      appendChild(child) {this.children.push(child);}};
  }
  for (const match of html.matchAll(/id="([^"]+)"/g)) nodes.set(match[1], element());
  nodes.get('gpuId').value = '2';
  nodes.get('meshTriangles').value = '250000';
  nodes.get('refinementTime').value = 'short';
  nodes.get('evaluationMethod').value = 'SuGaR';
  const catalog = {classes: Object.fromEntries(['Dewa','Dewi','Mitologi','Penabuh','Pengapit','Punakawan','Raksasa','Wanara'].map(cls => [cls, ['model_01','model_02']])), total_models: 18};
  catalog.classes.TanksAndTemples = ['tanks_and_temples_1','tanks_and_temples_2'];
  const status = Object.fromEntries(Object.entries(catalog.classes).map(([cls, models]) => [cls,
    Object.fromEntries(models.map(mdl => [mdl, {images_count: 0, overall_status: 'No Data', input_path: `datasets/${cls}/${mdl}/input/`, output_path: `outputs/{3dgs,sugar}/${cls}/${mdl}/`}]))]));
  const state = {jobs: [], counts: {}, paused: false, is_running: false, status: 'IDLE', elapsed_seconds: 0, logs: []};
  const context = vm.createContext({document: {getElementById: id => nodes.get(id) || null,
      createElement: element, querySelectorAll: () => [], addEventListener() {}},
    window: {addEventListener: (name, callback) => events[name] = callback, open() {}},
    navigator: {clipboard: {writeText: async () => {}}}, console,
    setTimeout() {}, setInterval() {},
    fetch: async (url, options) => {
      calls.push([url, options && options.body ? JSON.parse(options.body) : null]);
      const body = url === '/api/catalog' ? catalog : url === '/api/status' ? status : url === '/api/metrics' ? [] :
        url === '/api/queue' ? state : url.startsWith('/api/previews/') ? {images: []} : {added: ['id'], skipped_empty: [], duplicates: []};
      return {ok: true, json: async () => body};
    }});
  vm.runInContext(fs.readFileSync('web_dashboard/dashboard.js', 'utf8'), context);
  return {nodes, events, calls, context};
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
});

test('visual evaluation enqueues both methods while other work can be active', async () => {
  const app = page();
  await vm.runInContext("triggerRun('evaluate_visual', 'Dewa', 'model_01')", app.context);
  const requests = app.calls.filter(([url]) => url === '/api/run');
  assert.equal(requests.length, 2);
  assert.deepEqual(requests.map(([, body]) => body.evaluation_method), ['3DGS', 'SuGaR']);
  assert.equal(requests[0][1].gpu, 2);
  assert.equal(requests[0][1].resume, true);
});

test('geometry evaluation rejects absent scale before submitting a job', async () => {
  const app = page();
  await vm.runInContext("triggerRun('evaluate', 'Dewa', 'model_01')", app.context);
  assert.equal(app.calls.filter(([url]) => url === '/api/run').length, 0);
  assert.match(app.nodes.get('toast').textContent, /kalibrasi/);
});
