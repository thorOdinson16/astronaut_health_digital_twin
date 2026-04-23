// static/js/api.js
// ════════════════════════════════════════════════════════
// API LAYER
// ════════════════════════════════════════════════════════

function setApiStatus(state, msg, progress = null, runId = null) {
  const dot = document.getElementById('api-dot');
  const msgEl = document.getElementById('api-msg');
  const fill = document.getElementById('api-prog-fill');
  const ridEl = document.getElementById('api-run-id');
  
  if (dot) dot.className = state;
  if (msgEl) msgEl.textContent = msg;
  if (progress !== null && fill) fill.style.width = progress + '%';
  if (runId && ridEl) ridEl.textContent = runId;
}

async function apiPost(path, body) {
  const r = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

async function apiGet(path) {
  const r = await fetch(API_BASE + path);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

const _runConfigCache = new Map();
const _runConfigPending = new Map();

function _cacheRunConfig(runId, config) {
  if (!runId || !_hasConfig(config)) return;
  _runConfigCache.set(runId, config);
}

async function _ensureRunConfig(runId, fallbackConfig = null) {
  if (!runId) return null;
  if (_hasConfig(fallbackConfig)) {
    _cacheRunConfig(runId, fallbackConfig);
    return fallbackConfig;
  }
  if (_runConfigCache.has(runId)) return _runConfigCache.get(runId);
  if (_runConfigPending.has(runId)) return _runConfigPending.get(runId);
  
  const req = (async () => {
    try {
      const cfg = await apiGet(`/api/simulation/config/${runId}`);
      if (_hasConfig(cfg)) {
        _cacheRunConfig(runId, cfg);
        return cfg;
      }
      return null;
    } catch {
      return null;
    } finally {
      _runConfigPending.delete(runId);
    }
  })();
  
  _runConfigPending.set(runId, req);
  return req;
}

async function pollUntilComplete(runId, onProgress) {
  while (true) {
    let s;
    try {
      s = await apiGet(`/api/simulation/status/${runId}`);
    } catch (e) {
      const tracked = _simRuns.get(runId);
      const isCancelling = tracked && (tracked.status === 'cancelling' || tracked.status === 'cancelled');
      if (isCancelling && String(e.message || '').includes('404')) {
        const err = new Error('Simulation cancelled by user');
        err.code = 'SIM_CANCELLED';
        throw err;
      }
      throw e;
    }
    onProgress(s.progress || 0, s.status);
    if (s.status === 'completed') return s;
    if (s.status === 'failed') {
      const msg = s.error_message || 'unknown';
      if ((msg || '').toLowerCase().includes('manually stopped')) {
        const err = new Error('Simulation cancelled by user');
        err.code = 'SIM_CANCELLED';
        throw err;
      }
      throw new Error('Simulation failed: ' + msg);
    }
    await new Promise(r => setTimeout(r, 800));
  }
}

async function checkHealth() {
  try {
    const h = await apiGet('/api/health/');
    const bgStatus = document.getElementById('bg-status');
    const bgMode = document.getElementById('bg-mode');
    if (bgStatus) {
      bgStatus.textContent = 'ONLINE';
      bgStatus.className = 'sval G';
    }
    if (bgMode) bgMode.textContent = 'REST API';
  } catch(e) {
    const bgStatus = document.getElementById('bg-status');
    if (bgStatus) {
      bgStatus.textContent = 'OFFLINE';
      bgStatus.className = 'sval R';
    }
  }
}