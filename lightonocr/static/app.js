// LightOnOCR viewer. Served by `lightonocr viewer`; the routes are listed in server.py.
//
// One run is open at a time: the page image on the left with the layout blocks drawn over it, the
// transcription on the right. The model gives coordinates on a 0-1000 grid, so a box is a <div>
// positioned in CSS percentages (x / 10 %) inside the image's wrapper, and follows the image at any
// zoom. The sidebar lists the runs in out/ and the jobs creating new ones.

const $ = id => document.getElementById(id);

const api = async (url, options = {}) => {
  const r = await fetch(url, options);
  if (!r.ok) throw new Error((await r.text()).trim() || r.statusText);
  return r.status === 204 ? null : r.json();
};

// One colour per label. Page furniture is grey on purpose; labels not listed here get the light grey.
const COLORS = {
  title: '#d97706', section: '#b45309', text: '#2563eb', list: '#0891b2', caption: '#c026d3',
  footnote: '#ea580c', aside_text: '#4f46e5',
  header: '#78716c', footer: '#78716c', page_number: '#57534e',
  table: '#dc2626', chart: '#db2777', formula: '#0d9488', code: '#16a34a', image: '#7c3aed',
};
const colorOf = label => COLORS[label] || '#9ca3af';

const state = {
  runs: [],           // GET /api/runs
  jobs: [],           // GET /api/jobs
  run: null,          // the open run: {name, title, mode, pages: [{page, image, width, height, raw, blocks}]}
  page: 0,            // index into run.pages
  zoom: 1,            // 1 = the whole page fits its pane; larger zooms in and the pane scrolls
  raw: false,         // show the raw model output instead of rendering it
  hidden: new Set(),  // labels switched off with the chips
  file: null,         // file picked in the new-run dialog
  follow: null,       // id of the job to open as soon as it finishes
  grounding: true,    // whether the served model has a grounding mode (LightOnOCR-3 only)
  serving: null,      // the served model name, once the server has answered
};

// ---- Sidebar -----------------------------------------------------------------------------------

async function refreshSidebar() {
  [state.runs, state.jobs] = await Promise.all([api('/api/runs'), api('/api/jobs')]);
  renderSidebar();
  if (state.jobs.some(j => j.status === 'running')) watchJobs();
}

function renderSidebar() {
  const jobs = state.jobs.filter(j => j.status !== 'done');
  const runs = state.runs.filter(r => !jobs.some(j => j.name === r.name));  // a running job stands for its folder
  $('runs-count').textContent = runs.length || '';
  const nav = $('runs');
  nav.replaceChildren(...jobs.map(jobRow), ...runs.map(runRow));
  if (!nav.children.length) nav.innerHTML = '<div class="hint">No runs yet.</div>';
}

function runRow(run) {
  const row = document.createElement('a');
  row.className = 'run' + (state.run && state.run.name === run.name ? ' current' : '');
  row.href = '#' + encodeURIComponent(run.name);
  row.innerHTML = '<span class="run-name"></span><span class="run-meta"><i></i><span></span></span><button type="button" class="delete" title="Delete this run" aria-label="Delete">×</button>';
  row.querySelector('.run-name').textContent = run.name;
  row.querySelector('.run-name').title = run.title;
  row.querySelector('i').style.setProperty('--c', run.mode === 'grounding' ? COLORS.text : COLORS.page_number);
  row.querySelector('.run-meta span').textContent = `${run.mode} · ${run.page_count} page${run.page_count === 1 ? '' : 's'} · ${when(run.modified)}`;
  row.querySelector('.delete').onclick = e => { e.preventDefault(); deleteRun(e.target, run.name); };
  return row;
}

function jobRow(job) {
  const row = document.createElement('div');
  row.className = `run job ${job.status}`;
  row.innerHTML = '<span class="run-name"></span><span class="run-meta"><span></span></span><progress max="100"></progress><button type="button" class="delete" aria-label="Dismiss">×</button>';
  row.querySelector('.run-name').textContent = job.name;
  row.querySelector('.run-meta span').textContent = job.status === 'error' ? job.error : job.total ? `${job.done} / ${job.total} pages` : 'Rendering pages…';
  row.querySelector('progress').value = job.total ? Math.round(100 * job.done / job.total) : 0;
  if (job.status === 'error') {
    row.querySelector('progress').remove();
    row.querySelector('.delete').onclick = () => api('/api/jobs/' + job.id, {method: 'DELETE'}).then(refreshSidebar).catch(toast);
  } else {
    row.querySelector('.delete').remove();
  }
  return row;
}

// First click arms the button, a second click within 3 s deletes.
function deleteRun(button, name) {
  if (!button.classList.contains('armed')) {
    button.classList.add('armed');
    button.textContent = 'Delete?';
    setTimeout(() => { button.classList.remove('armed'); button.textContent = '×'; }, 3000);
    return;
  }
  api('/api/runs/' + encodeURIComponent(name), {method: 'DELETE'})
    .then(() => { if (state.run && state.run.name === name) closeRun(); return refreshSidebar(); })
    .catch(toast);
}

// "just now", "5 min ago", "3 h ago", "2 d ago", then the date.
function when(seconds) {
  const age = Date.now() / 1000 - seconds;
  if (age < 60) return 'just now';
  if (age < 3600) return `${Math.floor(age / 60)} min ago`;
  if (age < 86400) return `${Math.floor(age / 3600)} h ago`;
  if (age < 7 * 86400) return `${Math.floor(age / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined, {dateStyle: 'medium'});
}

// Poll the jobs while one is running; open the followed job's run when it is done.
let poller = null;
function watchJobs() {
  if (poller) return;
  poller = setInterval(async () => {
    const wasRunning = new Set(state.jobs.filter(j => j.status === 'running').map(j => j.id));
    try { state.jobs = await api('/api/jobs'); } catch (e) { return; }
    const finished = state.jobs.filter(j => wasRunning.has(j.id) && j.status === 'done');
    if (finished.length) state.runs = await api('/api/runs');
    renderSidebar();
    const followed = finished.find(j => j.id === state.follow);
    if (followed) { state.follow = null; openRun(followed.name).catch(toast); }
    if (!state.jobs.some(j => j.status === 'running')) { clearInterval(poller); poller = null; }
  }, 800);
}

// ---- Opening a run -----------------------------------------------------------------------------

async function openRun(name) {
  const run = await api('/api/runs/' + encodeURIComponent(name));
  Object.assign(state, {run, page: 0, zoom: 1, raw: false});
  state.hidden.clear();
  $('left').scrollTop = $('right').scrollTop = 0;
  if (decodeURIComponent(location.hash.slice(1)) !== name) history.replaceState(null, '', '#' + encodeURIComponent(name));
  document.body.classList.remove('menu-open');
  renderSidebar();
  renderViewer();
}

function closeRun() {
  state.run = null;
  history.replaceState(null, '', location.pathname);
  renderSidebar();
  renderViewer();
}

// ---- Viewer ------------------------------------------------------------------------------------

function renderViewer() {
  const run = state.run;
  document.title = run ? `${run.name} · LightOnOCR` : 'LightOnOCR';
  document.querySelector('.toolbar').classList.toggle('idle', !run);
  $('title').textContent = run ? run.title : '';
  $('mode').textContent = run ? run.mode : '';
  $('viewer').hidden = !run;
  $('empty').hidden = !!run;
  if (!run) return;

  const pages = run.pages, p = pages[state.page];
  $('raw').setAttribute('aria-pressed', String(state.raw));
  $('pageno').value = p.page;
  $('count').textContent = `/ ${pages.length}`;
  $('prev').disabled = state.page === 0;
  $('next').disabled = state.page === pages.length - 1;
  $('img').src = p.image;
  fitPage();

  const page = $('page'), panel = $('panel');
  page.querySelectorAll('.box').forEach(box => box.remove());
  $('filters').replaceChildren();
  panel.replaceChildren();
  spy.disconnect();

  if (state.raw) {
    const pre = document.createElement('pre');
    pre.className = 'raw';
    pre.textContent = p.raw || '';
    panel.appendChild(pre);
  }

  if (run.mode === 'grounding' && p.blocks) {
    // A box on the page and a card in the text pane for every block. Both carry data-i = index of
    // the block, which is what ties them together for hover and click.
    renderFilters(p.blocks);
    let shown = 0;
    p.blocks.forEach((block, i) => {
      if (state.hidden.has(block.label)) return;
      shown++;
      const [x1, y1, x2, y2] = block.bbox, c = colorOf(block.label), label = block.label + (block.continues ? '+' : '');
      const box = document.createElement('div');
      box.className = 'box' + (block.continues ? ' cont' : '');
      box.dataset.i = i;
      box.dataset.label = label;
      // Coordinates are on a 0-1000 grid, so x / 10 is a percentage of the image.
      box.style.cssText = `--c:${c};left:${x1 / 10}%;top:${y1 / 10}%;width:${(x2 - x1) / 10}%;height:${(y2 - y1) / 10}%`;
      page.appendChild(box);
      if (state.raw) return;  // boxes only; the text pane shows the raw output
      const card = document.createElement('div');
      card.className = 'blk';
      card.dataset.i = i;
      card.style.setProperty('--c', c);
      card.innerHTML = `<div class="meta"><span class="lab">${label}</span><span class="bbox">${block.bbox.join(', ')}</span></div>`;
      card.appendChild(renderMarkdown(block.text || ''));
      panel.appendChild(card);
      spy.observe(card);
    });
    if (!shown && !state.raw) panel.innerHTML = '<div class="empty">No blocks on this page</div>';
  } else if (!state.raw) {
    // Plain: the whole output rendered as one document.
    const doc = document.createElement('div');
    doc.className = 'doc';
    if (p.raw) doc.appendChild(renderMarkdown(p.raw)); else doc.innerHTML = '<div class="empty">Empty output</div>';
    panel.appendChild(doc);
  }
}

// The page is scaled so that it fits entirely in its pane, times the zoom factor. At zoom 1 the
// whole page is visible without scrolling; only the text pane scrolls.
function fitPage() {
  if (!state.run) return;
  const pane = $('left'), p = state.run.pages[state.page], pad = 50;  // 24px padding each side, 2px slack
  const scale = Math.min((pane.clientWidth - pad) / p.width, (pane.clientHeight - pad) / p.height);
  $('page').style.width = Math.max(80, Math.floor(p.width * scale * state.zoom)) + 'px';
  $('zoom-fit').textContent = state.zoom === 1 ? 'Fit' : state.zoom.toFixed(2).replace(/\.?0+$/, '') + '×';
  $('zoom-out').disabled = state.zoom <= 1;
  $('zoom-in').disabled = state.zoom >= 4;
}

function setZoom(zoom) {
  state.zoom = Math.min(4, Math.max(1, zoom));
  fitPage();
}

function goTo(index) {
  const pages = state.run ? state.run.pages : [];
  if (index < 0 || index >= pages.length || index === state.page) return;
  state.page = index;
  $('left').scrollTop = $('right').scrollTop = 0;
  renderViewer();
}

// One chip per label on the current page, with its count; click to hide or show the label.
function renderFilters(blocks) {
  const counts = {};
  blocks.forEach(block => { counts[block.label] = (counts[block.label] || 0) + 1; });
  Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([label, n]) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.style.setProperty('--c', colorOf(label));
    chip.setAttribute('aria-pressed', String(!state.hidden.has(label)));
    chip.title = 'Click to hide/show';
    chip.innerHTML = `<i></i>${label} <b>${n}</b>`;
    chip.onclick = () => { state.hidden.has(label) ? state.hidden.delete(label) : state.hidden.add(label); renderViewer(); };
    $('filters').appendChild(chip);
  });
}

// marked does not parse markdown inside raw HTML, and the model writes tables as HTML with
// markdown in the cells (e.g. **bold**), so bold the cell contents by hand before parsing.
const boldCells = text => text.replace(/<(td|th)([^>]*)>([^<]*)<\/\1>/g,
  (m, tag, attrs, inner) => `<${tag}${attrs}>${inner.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')}</${tag}>`);

// Markdown + LaTeX ($...$, $$...$$, \(...\), \[...\]) -> a .md element. Plain text if the CDN did not load.
function renderMarkdown(text) {
  const el = document.createElement('div');
  el.className = 'md';
  try { el.innerHTML = marked.parse(boldCells(text)); } catch (e) { el.textContent = text; }
  try {
    renderMathInElement(el, {
      delimiters: [
        {left: '$$', right: '$$', display: true}, {left: '\\[', right: '\\]', display: true},
        {left: '$', right: '$', display: false}, {left: '\\(', right: '\\)', display: false},
      ],
      throwOnError: false,
    });
  } catch (e) {}
  return el;
}

// Boxes whose card is visible in the text pane get .inview: the page shows where you are while you scroll.
const spy = new IntersectionObserver(entries => entries.forEach(entry => {
  const box = $('page').querySelector(`.box[data-i="${entry.target.dataset.i}"]`);
  if (box) box.classList.toggle('inview', entry.isIntersecting);
}), {root: $('right'), threshold: 0.3});

// Hover and click sync between a box on the page and its card in the text pane (same data-i).
const linked = i => document.querySelectorAll(`[data-i="${i}"]`);
document.addEventListener('mouseover', e => {
  const t = e.target.closest('[data-i]');
  if (t) linked(t.dataset.i).forEach(el => el.classList.add('on'));
});
document.addEventListener('mouseout', e => {
  const t = e.target.closest('[data-i]');
  if (t) linked(t.dataset.i).forEach(el => el.classList.remove('on'));
});
document.addEventListener('click', e => {  // click one side and the other side scrolls into view
  const t = e.target.closest('[data-i]');
  if (t) linked(t.dataset.i).forEach(el => { if (el !== t) el.scrollIntoView({block: 'center', behavior: 'smooth'}); });
});

// ---- Toolbar and keyboard -----------------------------------------------------------------------

$('prev').onclick = () => goTo(state.page - 1);
$('next').onclick = () => goTo(state.page + 1);
$('pageno').onchange = () => {  // jump to the page with that number, or the nearest one present
  const n = Number($('pageno').value), pages = state.run.pages;
  let best = 0;
  pages.forEach((p, i) => { if (Math.abs(p.page - n) < Math.abs(pages[best].page - n)) best = i; });
  goTo(best);
  $('pageno').value = pages[state.page].page;
  $('pageno').blur();
};
$('zoom-in').onclick = () => setZoom(state.zoom * 1.25);
$('zoom-out').onclick = () => setZoom(state.zoom / 1.25);
$('zoom-fit').onclick = () => setZoom(1);
$('raw').onclick = () => { state.raw = !state.raw; renderViewer(); };
$('menu').onclick = () => document.body.classList.toggle('menu-open');
window.addEventListener('resize', fitPage);

document.addEventListener('keydown', e => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName) || e.metaKey || e.ctrlKey || document.querySelector('dialog[open]')) return;
  if (e.key === 'n') openNewRun();
  if (!state.run) return;
  if (e.key === 'ArrowLeft') goTo(state.page - 1);
  else if (e.key === 'ArrowRight') goTo(state.page + 1);
  else if (e.key === 'r') $('raw').click();
  else if (e.key === '+' || e.key === '=') setZoom(state.zoom * 1.25);
  else if (e.key === '-') setZoom(state.zoom / 1.25);
  else if (e.key === '0') setZoom(1);
});

// ---- New run: pick a file (or drop one anywhere), POST it, follow the job -----------------------

function openNewRun(file = null) {
  pickFile(file);
  $('pages').value = '';
  $('form-error').hidden = true;
  offerGrounding();
  $('new-run-dialog').showModal();
}

function pickFile(file) {
  state.file = file;
  $('file-label').classList.toggle('has', !!file);
  $('file-text').textContent = file ? `${file.name} · ${(file.size / 1e6).toFixed(1)} MB` : 'Choose a PDF or image…';
  $('name').value = file ? file.name.replace(/\.[^.]+$/, '') : '';
}

// Grounding is a LightOnOCR-3 mode: with a LightOnOCR-1 or -2 server, only plain is offered.
function offerGrounding() {
  const form = $('new-run-form');
  form.querySelector('input[value=grounding]').disabled = !state.grounding;
  if (!state.grounding) form.mode.value = 'plain';
  const note = 'Grounding is available on LightOnOCR-3 only, not on LightOnOCR-1 and LightOnOCR-2.';
  $('grounding-note').textContent = state.grounding ? note : `${note} The server runs ${state.serving}: plain mode only.`;
  $('grounding-note').classList.toggle('warn', !state.grounding);
}

$('new-run').onclick = () => openNewRun();
$('cancel').onclick = () => $('new-run-dialog').close();
$('file').onchange = () => pickFile($('file').files[0]);
$('new-run-form').onsubmit = async e => {
  e.preventDefault();
  if (!state.file) { $('file-label').classList.add('has'); $('file-text').textContent = 'Pick a file first'; return; }
  const form = e.target;
  const params = new URLSearchParams({name: $('name').value, mode: form.mode.value, pages: $('pages').value.trim()});
  $('submit').disabled = true;
  try {
    const job = await api('/api/runs?' + params, {method: 'POST', body: state.file, headers: {'X-Filename': encodeURIComponent(state.file.name)}});
    state.follow = job.id;
    $('new-run-dialog').close();
    await refreshSidebar();
  } catch (err) {
    $('form-error').textContent = err.message;
    $('form-error').hidden = false;
  }
  $('submit').disabled = false;
};

// Drop a file anywhere on the page.
const hasFiles = e => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
let dragDepth = 0;
window.addEventListener('dragenter', e => { if (hasFiles(e)) { dragDepth++; $('dropzone').hidden = false; } });
window.addEventListener('dragleave', e => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; $('dropzone').hidden = true; } });
window.addEventListener('dragover', e => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener('drop', e => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $('dropzone').hidden = true;
  if (e.dataTransfer.files[0]) openNewRun(e.dataTransfer.files[0]);
});

// ---- Model server: status at the bottom of the sidebar, changed in a dialog, kept in out/settings.json --

function renderEndpoint(s) {
  state.grounding = s.grounding !== false;
  state.serving = s.serving;
  offerGrounding();
  $('endpoint-url').textContent = (s.serving ? s.serving + ' @ ' : '') + s.base_url;
  $('endpoint-dot').className = 'dot ' + (s.reachable ? 'ok' : 'down');
  $('endpoint').title = s.reachable ? `Model server online, serving ${s.models.join(', ')}` : `Model server unreachable: ${s.error}`;
}

function showEndpointStatus(s) {
  const el = $('endpoint-status');
  el.className = 'status ' + (s.reachable ? 'ok' : 'down');
  el.textContent = s.reachable ? `Online, serving ${s.models.join(', ')}` : `Unreachable: ${s.error}. Saved anyway; runs will fail until it answers.`;
}

// GET /api/settings probes the server, which takes a few seconds when it is down.
const probeEndpoint = () => api('/api/settings').then(s => { renderEndpoint(s); return s; });

function openSettings() {
  $('endpoint-status').className = 'status';
  $('endpoint-status').textContent = 'Checking…';
  $('settings-dialog').showModal();
  probeEndpoint().then(s => { $('base-url').value = s.base_url; $('model-name').value = s.model || ''; showEndpointStatus(s); }).catch(toast);
}

$('endpoint').onclick = openSettings;
$('settings-cancel').onclick = () => $('settings-dialog').close();
$('settings-form').onsubmit = async e => {
  e.preventDefault();
  $('settings-save').disabled = true;
  $('endpoint-status').className = 'status';
  $('endpoint-status').textContent = 'Saving and checking…';
  try {
    const body = JSON.stringify({base_url: $('base-url').value.trim(), model: $('model-name').value.trim()});
    const s = await api('/api/settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body});
    renderEndpoint(s);
    showEndpointStatus(s);
    if (s.reachable) $('settings-dialog').close();
  } catch (err) {
    $('endpoint-status').className = 'status down';
    $('endpoint-status').textContent = err.message;
  }
  $('settings-save').disabled = false;
};

// ---- Errors, routing, boot ----------------------------------------------------------------------

let toastTimer = null;
function toast(err) {
  console.error(err);
  $('toast').textContent = err.message || String(err);
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 5000);
}

window.addEventListener('focus', () => refreshSidebar().catch(() => {}));  // pick up runs made by the CLI meanwhile

window.addEventListener('hashchange', () => {
  const name = decodeURIComponent(location.hash.slice(1));
  if (name && (!state.run || state.run.name !== name)) openRun(name).catch(toast);
});

async function boot() {
  api('/api/info').then(info => { $('info').textContent = info.out; $('endpoint-url').textContent = info.base_url; }).catch(() => {});
  probeEndpoint().catch(() => {});
  await refreshSidebar();
  const wanted = decodeURIComponent(location.hash.slice(1));
  if (wanted && state.runs.some(r => r.name === wanted)) await openRun(wanted);
  else if (state.runs.length) await openRun(state.runs[0].name);
  else renderViewer();
}

boot().catch(toast);
