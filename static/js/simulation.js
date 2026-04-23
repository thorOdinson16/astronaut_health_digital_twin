// static/js/simulation.js
// ════════════════════════════════════════════════════════
// SIMULATION LIFECYCLE
// ════════════════════════════════════════════════════════

let currentRunId = null;
let simResults = null;
let simSummary = null;
let simEvents = null;
let currentAnalytics = null;

function onSimReady() {
  document.getElementById('tl-section').style.display = 'block';
  document.getElementById('mission-risk-section').style.display = 'none';
  document.getElementById('mc-section').style.display = 'none';

  const compare = document.getElementById('run-compare-section');
  if (compare) compare.style.display = 'none';

  const explain = document.getElementById('risk-explain-wrap');
  if (explain) { explain.style.display = 'none'; explain.textContent = ''; }

  const actions = document.getElementById('risk-actions');
  if (actions) actions.style.display = 'none';

  initCharts();

  document.getElementById('mission-risk-stats').innerHTML = '';
  document.getElementById('rtable').innerHTML = '';

  const conc = document.getElementById('mission-risk-conclusions');
  if (conc) { conc.style.display = 'none'; conc.innerHTML = ''; }
}

function onPlaybackComplete() {
  document.getElementById('mission-risk-section').style.display = 'block';
  const actions = document.getElementById('risk-actions');
  if (actions) actions.style.display = currentAnalytics ? 'flex' : 'none';
  if (!currentAnalytics && currentRunId) {
    loadAnalytics(currentRunId).finally(() => {
      if (actions && currentAnalytics) actions.style.display = 'flex';
    });
  }
  renderRunComparison();
}

async function startSimulation(evt) {
  if (evt && typeof evt.preventDefault === 'function') evt.preventDefault();
  setApiStatus('busy', 'Submitting simulation to server...', 0);

  const cfg = {
    mission_duration_hours: +document.getElementById('ctrl-cyc').value * 24,
    time_step_minutes: 30,
    baseline_hr: 75,
    baseline_sleep_quality: 0.80,
    initial_fatigue: 0.0,
    sms_rate_per_day: +document.getElementById('ctrl-lam').value / 100,
    recovery_factor: +document.getElementById('ctrl-rec').value / 10,
    fatigue_sensitivity: +document.getElementById('ctrl-fsens').value / 10,
    enable_motion_sickness: true,
    enable_sleep_disruption: true,
    use_biogears: true,
    save_trajectories: true,
    save_events: true,
  };

  let runId = null;
  try {
    markUserInteraction();
    const resp = await apiPost('/api/simulation/run', cfg);
    runId = resp.run_id;
    _simTrackAdd(runId);
    _cacheRunConfig(runId, cfg);

    const trackedRun = _simRuns.get(runId);
    if (trackedRun) trackedRun.config = cfg;
    _showSimRunParams(runId, 'selected');

    document.getElementById('bg-cyc').textContent = runId;
    document.getElementById('api-run-id').textContent = runId;
    setApiStatus('busy', `Running: ${runId}`, 0, runId);

    await pollUntilComplete(runId, (pct, status) => {
      setApiStatus('busy', `[${status.toUpperCase()}] ${pct.toFixed(0)}%`, pct, runId);
      document.getElementById('api-prog-fill').style.width = pct + '%';
      _simTrackUpdate(runId, pct, status.toUpperCase());
    });

    _simTrackUpdate(runId, 99, 'LOADING');
    setApiStatus('busy', 'Fetching results...', 99, runId);
    await fetchAllResults(runId);

    currentRunId = runId;
    _simTrackDone(runId);
    setApiStatus('', `Results loaded — ${simResults.state.time.length} timesteps`, 100, runId);

    document.getElementById('btn-play').disabled = false;
    document.getElementById('btn-reset').disabled = false;

    onSimReady();
    buildTimeline();
    populateStatusPanel();
    checkHealth();

  } catch (e) {
    if (runId && e && e.code === 'SIM_CANCELLED') {
      _simTrackCancelled(runId);
      setApiStatus('', `Simulation cancelled — ${runId}`, 0, runId);
      return;
    }
    if (runId) _simTrackError(runId, e.message);
    setApiStatus('err', 'Error: ' + e.message, 0);
    console.error(e);
  }
}

async function fetchAllResults(runId) {
  currentAnalytics = null;
  const [results, summary, events] = await Promise.all([
    apiGet(`/api/data/results/${runId}`),
    apiGet(`/api/data/results/${runId}/summary`),
    apiGet(`/api/data/results/${runId}/events`),
  ]);
  simResults = results;
  simSummary = summary;
  simEvents = events.events || [];

  computeFullChartData();
  await loadAnalytics(runId);
}

function populateStatusPanel() {
  if (!simSummary || !simSummary.metrics || !simSummary.risk_assessment) return;
  const m = simSummary.metrics;

  document.getElementById('st-pf').textContent  = m.fatigue.peak.toFixed(3);
  document.getElementById('st-sms').textContent = m.motion_sickness.episodes;
  document.getElementById('st-dis').textContent = '—';
  document.getElementById('st-sq').textContent  = m.sleep_quality.mean.toFixed(2);
  document.getElementById('st-hr').textContent  = m.heart_rate.mean.toFixed(0) + ' bpm';

  const elog = document.getElementById('elog');
  elog.innerHTML = '';
  if (!simEvents.length) {
    elog.innerHTML = '<div class="eitem" style="color:rgba(120,168,200,.3)">No events recorded</div>';
    return;
  }
  simEvents.slice(0, 10).forEach(e => {
    const isSms = e.type && e.type.toLowerCase().includes('motion');
    const isDis = e.type && e.type.toLowerCase().includes('sleep');
    const cls   = isSms ? 'S' : isDis ? 'D' : '';
    const t     = (e.simulation_time || 0).toFixed(1);
    const sev   = e.severity ? ` sev=${(+e.severity).toFixed(2)}` : '';
    const el    = document.createElement('div');
    el.className = `eitem ${cls}`;
    el.textContent = `[T+${t}h] ${e.type || 'EVENT'}${sev}`;
    elog.appendChild(el);
  });
}

function showMC() {
  if (!simResults || !simSummary) return;
  const m = simSummary.metrics;
  const ra = simSummary.risk_assessment;

  const stats = [
    ['PEAK FATIGUE',     m.fatigue.peak.toFixed(2),             m.fatigue.peak > 7 ? 'R' : m.fatigue.peak > 5 ? 'W' : ''],
    ['MEAN SLEEP QUALITY', m.sleep_quality.mean.toFixed(2),     m.sleep_quality.mean < .4 ? 'R' : m.sleep_quality.mean < .6 ? 'W' : ''],
    ['SMS EPISODES',     m.motion_sickness.episodes,            m.motion_sickness.episodes > 4 ? 'R' : m.motion_sickness.episodes > 2 ? 'W' : ''],
    ['COMPOSITE RISK',   ra ? ra.composite_risk : '—',          ra && ra.composite_risk === 'HIGH' ? 'R' : ra && ra.composite_risk === 'MEDIUM' ? 'W' : ''],
  ];

  document.getElementById('mission-risk-stats').innerHTML = stats.map(([l, v, c]) =>
    `<div class="mcstat"><div class="mcslabel">${l}</div><div class="mcsval ${c}">${v}</div></div>`
  ).join('');

  const actions = document.getElementById('risk-actions');
  if (actions) actions.style.display = 'flex';
}

function showReport() {
  if (!simSummary) return;
  const m = simSummary.metrics;
  const ra = simSummary.risk_assessment;

  const rows = [
    ['OVERALL RISK LEVEL',   currentAnalytics?.risk_report?.overall_risk_level || '—',        currentAnalytics?.risk_report?.overall_risk_level === 'CRITICAL' ? 'R' : currentAnalytics?.risk_report?.overall_risk_level === 'HIGH' ? 'R' : currentAnalytics?.risk_report?.overall_risk_level === 'MODERATE' ? 'W' : ''],
    ['PEAK FATIGUE INDEX',   m.fatigue.peak.toFixed(3),                                         m.fatigue.peak > 7 ? 'R' : m.fatigue.peak > 5 ? 'W' : ''],
    ['MEAN SLEEP QUALITY',   m.sleep_quality.mean.toFixed(2),                                   m.sleep_quality.mean < .4 ? 'R' : m.sleep_quality.mean < .6 ? 'W' : ''],
    ['SLEEP EFFICIENCY',     m.sleep_quality.efficiency.toFixed(1) + '%',                       m.sleep_quality.efficiency < 40 ? 'R' : m.sleep_quality.efficiency < 60 ? 'W' : ''],
    ['SMS EPISODES',         m.motion_sickness.episodes,                                        m.motion_sickness.episodes > 4 ? 'R' : m.motion_sickness.episodes > 2 ? 'W' : ''],
    ['AVG HEART RATE',       m.heart_rate.mean.toFixed(0) + ' bpm',                            m.heart_rate.mean > 100 ? 'W' : ''],
    ['HR VARIABILITY',       m.heart_rate.variability.toFixed(1) + ' bpm',                     ''],
    ['PEAK FATIGUE RISK',    ra ? ra.fatigue_risk : '—',                                        ra && ra.fatigue_risk === 'HIGH' ? 'R' : ra && ra.fatigue_risk === 'MEDIUM' ? 'W' : ''],
    ['PEAK SLEEP RISK',      ra ? ra.sleep_risk : '—',                                          ra && ra.sleep_risk === 'HIGH' ? 'R' : ra && ra.sleep_risk === 'MEDIUM' ? 'W' : ''],
    ['MISSION DURATION',     simSummary.duration_hours.toFixed(0) + ' hours',                  ''],
  ];

  document.getElementById('rtable').innerHTML = rows.map(([l, v, c]) =>
    `<tr class="${c}"><td style="padding:6px 10px;border-bottom:1px solid var(--border)">${l}</td>` +
    `<td style="padding:6px 10px;border-bottom:1px solid var(--border);text-align:right;color:var(--cyan)">${v}</td></tr>`
  ).join('');
}

// ════════════════════════════════════════════════════════
// SIMULATION TRACKER — ℹ overlay when runs are active
// ════════════════════════════════════════════════════════

const _simRuns = new Map();
let _simTooltipPreviewRunId = null;
let _hideTooltipTimer = null;
let _elapsedTimer = null;

function _simPhaseLabel(pct) {
  if (pct <  5) return 'INITIALIZING';
  if (pct < 35) return 'PHYSIOLOGICAL SIM';
  if (pct < 65) return 'FATIGUE & SLEEP';
  if (pct < 88) return 'EVENTS & STRESSORS';
  if (pct < 99) return 'FINALIZING';
  return 'LOADING RESULTS';
}

function _simPhaseColor(status) {
  if (status === 'done') return 'var(--green)';
  if (status === 'cancelled' || status === 'cancelling') return 'var(--amber)';
  if (status === 'err') return 'var(--red)';
  return 'var(--amber)';
}

function _simFillClass(status) {
  if (status === 'done') return 'done';
  if (status === 'err') return 'err';
  if (status === 'cancelled' || status === 'cancelling') return 'cancelled';
  return '';
}

function _simStatusLabel(run) {
  if (run.status === 'done') return 'COMPLETE';
  if (run.status === 'err') return 'ERROR';
  if (run.status === 'cancelled') return 'CANCELLED';
  if (run.status === 'cancelling') return 'CANCELLING';
  return _simPhaseLabel(run.pct);
}

function _simSubLabel(run) {
  if (run.status === 'done') return 'RESULTS READY';
  if (run.status === 'err') return 'FAILED';
  if (run.status === 'cancelled') return 'STOPPED BY USER';
  if (run.status === 'cancelling') return 'STOP REQUESTED';
  return _simPhaseLabel(run.pct);
}

function _isTrackedRunActive(status) {
  return status === 'running' || status === 'cancelling';
}

function _entryId(runId) {
  return 'sr-' + runId.replace(/[^a-z0-9]/gi, '_');
}

function _animateEntryOut(el) {
  if (!el || el.classList.contains('leaving')) return;
  el.classList.add('leaving');
  setTimeout(() => el.remove(), 210);
}

function _schedulePruneRun(runId, delayMs) {
  setTimeout(() => {
    const el = document.getElementById(_entryId(runId));
    if (el) _animateEntryOut(el);
    setTimeout(() => {
      _simRuns.delete(runId);
      _updateRunButton();
      _renderTooltip();
      if (_simRuns.size === 0) hideSimTooltip();
    }, 220);
  }, delayMs);
}

function _setTooltipVisible(visible) {
  const tip = document.getElementById('sim-tooltip');
  if (!tip) return;
  tip.classList.toggle('visible', !!visible);
}

function _updateRunButton() {
  const icon = document.getElementById('sim-info-icon');
  const btn  = document.getElementById('btn-run');
  if (!icon || !btn) return;

  const active = [..._simRuns.values()].filter(r => _isTrackedRunActive(r.status)).length;
  const hasTrackedRuns = _simRuns.size > 0;

  if (hasTrackedRuns) {
    icon.classList.add('visible');
    icon.innerHTML = active > 1
      ? `ℹ<sup style="font-size:6px;vertical-align:super;color:var(--amber)">${active}</sup>`
      : 'ℹ';
    btn.style.paddingRight = '30px';
  } else {
    icon.classList.remove('visible');
    btn.style.paddingRight = '';
    hideSimTooltip();
  }
}

function showSimTooltip() {
  cancelHideSimTooltip();
  _renderTooltip();
  _setTooltipVisible(true);
  clearInterval(_elapsedTimer);
  _elapsedTimer = setInterval(_updateElapsed, 1000);
}

function hideSimTooltip() {
  _setTooltipVisible(false);
  clearInterval(_elapsedTimer);
}

function scheduleHideSimTooltip() {
  _hideTooltipTimer = setTimeout(hideSimTooltip, 220);
}

function cancelHideSimTooltip() {
  clearTimeout(_hideTooltipTimer);
}

function _updateRunEntry(runId) {
  const r = _simRuns.get(runId);
  if (!r) return;

  const eid = _entryId(runId);
  const barEl    = document.getElementById(`${eid}-bar`);
  const phaseEl  = document.getElementById(`${eid}-phase`);
  const lblEl    = document.getElementById(`${eid}-lbl`);
  const metaEl   = document.getElementById(`${eid}-meta`);
  const cancelEl = document.getElementById(`${eid}-cancel`);
  const paramsEl = document.getElementById(`${eid}-params`);
  const entryEl  = document.getElementById(eid);

  const phaseText  = _simStatusLabel(r);
  const phaseColor = _simPhaseColor(r.status);
  const fillCls    = _simFillClass(r.status);
  const elapsed    = ((Date.now() - r.startTs) / 1000).toFixed(0);

  if (barEl)   { barEl.style.width = r.pct + '%'; barEl.className = 'sim-prog-fill ' + fillCls; }
  if (entryEl) entryEl.classList.toggle('cancelling', r.status === 'cancelling');
  if (phaseEl) { phaseEl.textContent = phaseText; phaseEl.style.color = phaseColor; }
  if (lblEl)   lblEl.textContent = _simSubLabel(r);
  if (metaEl)  metaEl.textContent = `${elapsed}s · ${r.pct.toFixed(0)}%`;
  if (paramsEl && _hasConfig(r.config)) paramsEl.textContent = _formatRunListParamLine(r.config);

  if (cancelEl) {
    const canCancel = r.status === 'running';
    const cancelling = r.status === 'cancelling';
    cancelEl.disabled = !(canCancel || cancelling);
    cancelEl.textContent = cancelling ? '...' : 'CANCEL';
    cancelEl.style.visibility = (canCancel || cancelling) ? 'visible' : 'hidden';
  }
}

function _renderTooltip() {
  const list = document.getElementById('sim-tooltip-list');
  if (!list) return;

  if (_simRuns.size === 0) {
    list.innerHTML = '<div style="color:rgba(120,168,200,.35);text-align:center;padding:4px 0;letter-spacing:1px">NO ACTIVE RUNS</div>';
    return;
  }

  _simRuns.forEach((r, id) => {
    const eid  = _entryId(id);
    const short = id.length > 22 ? id.slice(0, 10) + '…' + id.slice(-8) : id;

    let entry = document.getElementById(eid);
    if (!entry) {
      entry = document.createElement('div');
      entry.id = eid;
      entry.className = 'sim-run-entry';
      entry.innerHTML = `
        <div class="sim-run-header">
          <span class="sim-run-id-label" title="${id}">${short}</span>
          <div class="sim-run-actions">
            <button type="button" class="sim-run-action cancel" id="${eid}-cancel" onclick="cancelSimulationRun('${id}', event)">CANCEL</button>
            <span class="sim-run-phase-label" id="${eid}-phase"></span>
          </div>
        </div>
        <div class="sim-prog-track"><div class="sim-prog-fill" id="${eid}-bar"></div></div>
        <div style="font-size:7px;color:rgba(0,212,255,.5);letter-spacing:.5px;margin-top:3px;min-height:10px" id="${eid}-params"></div>
        <div class="sim-run-sub"><span id="${eid}-lbl"></span><span id="${eid}-meta"></span></div>`;
      list.appendChild(entry);
    }

    if (!_hasConfig(r.config)) {
      _ensureRunConfig(id).then(cfg => {
        if (cfg) {
          const updated = _simRuns.get(id);
          if (updated) updated.config = cfg;
          _updateRunEntry(id);
        }
      });
    }
    _updateRunEntry(id);
  });

  list.querySelectorAll('.sim-run-entry').forEach(el => {
    const stillTracked = [..._simRuns.keys()].some(k => _entryId(k) === el.id);
    if (!stillTracked) _animateEntryOut(el);
  });
}

function _updateElapsed() {
  const tip = document.getElementById('sim-tooltip');
  if (!tip || !tip.classList.contains('visible')) return;
  _simRuns.forEach((_, id) => _updateRunEntry(id));
}

// ── Lifecycle ─────────────────────────────────────────────────────────────
function _simTrackAdd(runId) {
  _simRuns.set(runId, { pct: 0, phase: 'STARTING', startTs: Date.now(), status: 'running', config: null });
  _simTooltipPreviewRunId = runId;
  _updateRunButton();
  _renderTooltip();
}

function _simTrackUpdate(runId, pct, phase) {
  const r = _simRuns.get(runId);
  if (!r) return;
  r.pct = Math.max(0, Math.min(100, +pct || 0));
  if (phase) r.phase = String(phase).toUpperCase();
  if (r.status !== 'cancelling') r.status = 'running';
  _updateRunEntry(runId);
  _updateRunButton();
}

function _simTrackDone(runId) {
  const r = _simRuns.get(runId);
  if (!r) return;
  r.pct = 100; r.status = 'done';
  _updateRunEntry(runId);
  _updateRunButton();
  _schedulePruneRun(runId, 5500);
}

function _simTrackError(runId, msg) {
  const r = _simRuns.get(runId);
  if (!r) return;
  r.status = 'err';
  r.phase = (msg || 'Error').slice(0, 50).toUpperCase();
  _updateRunEntry(runId);
  _updateRunButton();
  _schedulePruneRun(runId, 7000);
}

function _simTrackCancelling(runId) {
  const r = _simRuns.get(runId);
  if (!r) return;
  r.status = 'cancelling'; r.phase = 'STOP REQUESTED';
  _updateRunEntry(runId);
  _updateRunButton();
}

function _simTrackCancelled(runId) {
  const r = _simRuns.get(runId);
  if (!r) return;
  r.status = 'cancelled'; r.phase = 'STOPPED BY USER';
  _updateRunEntry(runId);
  _updateRunButton();
  _schedulePruneRun(runId, 4500);
}

async function cancelSimulationRun(runId, evt) {
  if (evt && typeof evt.stopPropagation === 'function') evt.stopPropagation();
  const r = _simRuns.get(runId);
  if (!r || r.status !== 'running') return;
  if (!confirm(`Cancel simulation "${runId}"?`)) return;

  _simTrackCancelling(runId);
  setApiStatus('busy', `Cancelling ${runId}...`, r.pct, runId);

  try {
    const res = await fetch(API_BASE + `/api/simulation/stop/${runId}`, { method: 'POST' });
    if (!res.ok && res.status !== 404) throw new Error(`POST /api/simulation/stop/${runId} → ${res.status}`);
  } catch (e) {
    r.status = 'running'; r.phase = 'RESUMED';
    _updateRunEntry(runId);
    _updateRunButton();
    setApiStatus('err', 'Cancel failed: ' + e.message, r.pct, runId);
    alert('Cancel failed: ' + e.message);
  }
}

// ── Show run params in sim tooltip ────────────────────────────────────────
async function _showSimRunParams(runId, source = 'hover') {
  _simTooltipPreviewRunId = runId;
  const run = _simRuns.get(runId);
  if (!run) return;

  const cfg = await _ensureRunConfig(runId, run.config || null);
  if (_simTooltipPreviewRunId !== runId) return;
  if (cfg) run.config = cfg;
  _updateRunEntry(runId);
}

// ════════════════════════════════════════════════════════
// LOAD MODAL
// ════════════════════════════════════════════════════════

let _selectedLoadRunId = null;
let _loadPreviewRunId  = null;
let allRuns = [];

function _normalizeRunStatus(status) {
  const s = String(status || 'completed').toLowerCase();
  if (s === 'pending' || s === 'running' || s === 'completed' || s === 'failed') return s;
  if (s === 'queued' || s === 'started') return 'running';
  return 'completed';
}

function _formatRunListParamLine(cfg) {
  if (!_hasConfig(cfg)) return '—';
  const days = cfg.mission_duration_hours !== undefined ? `${_compactNumber((Number(cfg.mission_duration_hours) || 0) / 24, 0)}d` : null;
  const dt   = cfg.time_step_minutes !== undefined      ? `${_compactNumber(cfg.time_step_minutes, 0)}m step` : null;
  const sms  = cfg.sms_rate_per_day !== undefined       ? `${_compactNumber(cfg.sms_rate_per_day, 2)} sms` : null;
  const rec  = cfg.recovery_factor !== undefined        ? `${_compactNumber(cfg.recovery_factor, 1)} rf` : null;
  const fs   = cfg.fatigue_sensitivity !== undefined    ? `${_compactNumber(cfg.fatigue_sensitivity, 1)} fs` : null;
  return [days, dt, sms, rec, fs].filter(Boolean).join(' · ') || '—';
}

function _getRunById(runId) {
  return allRuns.find(r => r.run_id === runId) || null;
}

function _setLoadModalSelectionState(run) {
  const loadBtn = document.getElementById('btn-load-confirm');
  const delBtn  = document.getElementById('btn-delete-run');
  if (!loadBtn || !delBtn) return;
  if (!run) {
    loadBtn.disabled = true;
    loadBtn.textContent = '◈ LOAD SELECTED';
    delBtn.disabled = true;
    delBtn.textContent = '✕ DELETE';
    return;
  }
  const canLoad = run.status === 'completed';
  loadBtn.disabled = !canLoad;
  loadBtn.textContent = canLoad ? '◈ LOAD SELECTED' : '◈ LOAD (COMPLETED ONLY)';
  delBtn.disabled = false;
  delBtn.textContent = '✕ DELETE';
}

async function openLoadModal() {
  document.getElementById('load-overlay').classList.add('open');
  _selectedLoadRunId = null;
  _loadPreviewRunId  = null;
  _setLoadModalSelectionState(null);
  await refreshRunList();
}

function closeLoadModal(e) {
  if (e && e.target !== document.getElementById('load-overlay')) return;
  document.getElementById('load-overlay').classList.remove('open');
  _selectedLoadRunId = null;
  _loadPreviewRunId  = null;
  const confirmBtn = document.getElementById('btn-load-confirm');
  const deleteBtn  = document.getElementById('btn-delete-run');
  if (confirmBtn) confirmBtn.disabled = true;
  if (deleteBtn)  deleteBtn.disabled  = true;
}

async function refreshRunList() {
  const spinner = document.getElementById('load-refresh-spinner');
  const list    = document.getElementById('load-list');
  if (spinner) spinner.textContent = '◌ FETCHING RUNS...';
  if (list)    list.innerHTML = '';
  _selectedLoadRunId = null;
  _loadPreviewRunId  = null;
  _setLoadModalSelectionState(null);

  try {
    let runs = [];
    try {
      const r = await apiGet('/api/simulation/list?limit=20');
      runs = r.runs || r || [];
    } catch {
      try { const r = await apiGet('/api/data/runs');       runs = r.runs || r || []; }
      catch { const r = await apiGet('/api/simulation/runs'); runs = r.runs || r || []; }
    }

    allRuns = (runs || []).map(run => {
      if (typeof run === 'string') return { run_id: run, status: 'completed', config: {}, created_at: run };
      const runId = run.run_id || run.id;
      if (!runId) return null;
      return { run_id: runId, status: _normalizeRunStatus(run.status), config: run.config || {}, created_at: run.created_at || run.started_at || run.completed_at || runId };
    }).filter(Boolean);

    allRuns.forEach(run => _cacheRunConfig(run.run_id, run.config));

    const counts = { pending: 0, running: 0, completed: 0, failed: 0 };
    allRuns.forEach(r => { counts[r.status] = (counts[r.status] || 0) + 1; });
    if (spinner) spinner.textContent = `${allRuns.length} runs • ${counts.running || 0} running • ${counts.completed || 0} completed`;

    if (!allRuns.length) {
      if (list) list.innerHTML = '<div id="load-empty">NO RUNS FOUND</div>';
      return;
    }

    allRuns.sort((a, b) => {
      const da = _parseLikeDate(a.created_at), db = _parseLikeDate(b.created_at);
      if (da && db) return db.getTime() - da.getTime();
      return String(b.created_at || b.run_id || '').localeCompare(String(a.created_at || a.run_id || ''));
    });

    const preloadTasks = [];
    allRuns.forEach(run => {
      const id = run.run_id;
      const cfg = run.config || {};
      const status = _normalizeRunStatus(run.status);
      const params = _formatRunListParamLine(cfg);

      if (!_hasConfig(cfg)) {
        preloadTasks.push(_ensureRunConfig(id).then(fullCfg => {
          if (!fullCfg) return;
          run.config = fullCfg;
          const row = list?.querySelector(`.run-item[data-run-id="${id}"]`);
          if (!row) return;
          const summary = row.querySelector('[data-role="run-summary"]');
          if (summary) summary.textContent = _formatRunListParamLine(fullCfg);
        }));
      }

      const ts = run.created_at ? new Date(run.created_at).toLocaleString() : '';
      const div = document.createElement('div');
      div.className = 'run-item';
      div.dataset.runId = id;
      div.dataset.status = status;
      div.innerHTML = `
        <div style="flex:1;min-width:0">
          <div class="run-id">${id}</div>
          <div style="font-size:8px;color:rgba(0,212,255,.6);letter-spacing:.5px;margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" data-role="run-summary">${params || '—'}</div>
          <div class="run-meta" style="margin-top:2px;color:rgba(120,168,200,.45)">${ts || '—'}${status !== 'completed' ? ' · unavailable' : ''}</div>
        </div>
        <span class="run-badge ${status}">${status.toUpperCase()}</span>`;
      div.onclick = () => selectRun(id, div);
      if (list) list.appendChild(div);
    });

    if (preloadTasks.length) Promise.allSettled(preloadTasks);

  } catch (e) {
    if (spinner) spinner.textContent = '';
    if (list) list.innerHTML = `<div id="load-empty" style="color:var(--red)">ERROR: ${e.message}</div>`;
  }
}

function selectRun(id, el) {
  document.querySelectorAll('.run-item').forEach(r => r.classList.remove('selected'));
  el.classList.add('selected');
  _selectedLoadRunId = id;
  const run = _getRunById(id);
  _setLoadModalSelectionState(run);
}

// Keep old name as alias for backwards compat with load-modal HTML onclick
function selectLoadRun(runId) {
  const el = document.querySelector(`.run-item[data-run-id="${runId}"]`);
  if (el) selectRun(runId, el);
}

async function deleteSelectedRun() {
  const runId = _selectedLoadRunId;
  if (!runId) return;
  const run = _getRunById(runId);
  const isActive = run && (run.status === 'running' || run.status === 'pending');
  const q = isActive
    ? `Run "${runId}" is ${run.status.toUpperCase()}. It will be cancelled before delete. Continue?`
    : `Delete run "${runId}"? This cannot be undone.`;
  if (!confirm(q)) return;

  const btn     = document.getElementById('btn-delete-run');
  const loadBtn = document.getElementById('btn-load-confirm');
  if (btn)     { btn.disabled = true; btn.textContent = '⟳ DELETING...'; }
  if (loadBtn) loadBtn.disabled = true;

  try {
    if (isActive) {
      const tracked = _simRuns.get(runId);
      if (tracked) _simTrackCancelling(runId);
      const stopRes = await fetch(API_BASE + `/api/simulation/stop/${runId}`, { method: 'POST' });
      if (!stopRes.ok && stopRes.status !== 404) throw new Error(`POST /api/simulation/stop/${runId} → ${stopRes.status}`);
      await new Promise(r => setTimeout(r, 250));
    }

    const delRes = await fetch(API_BASE + `/api/simulation/delete/${runId}`, { method: 'DELETE' });
    if (!delRes.ok) throw new Error(`DELETE /api/simulation/delete/${runId} → ${delRes.status}`);

    if (_simRuns.has(runId)) _simTrackCancelled(runId);
    _selectedLoadRunId = null;
    _setLoadModalSelectionState(null);
    setApiStatus('', `Deleted run ${runId}`, 0);
    await refreshRunList();

  } catch (e) {
    alert('Delete failed: ' + e.message);
    _setLoadModalSelectionState(run);
    if (btn) btn.textContent = '✕ DELETE';
  }
}

async function confirmLoad() {
  const runId = _selectedLoadRunId;
  if (!runId) return;
  const selectedRun = _getRunById(runId);
  if (selectedRun && selectedRun.status !== 'completed') {
    alert(`Run ${runId} is ${selectedRun.status.toUpperCase()}. Only completed runs can be loaded.`);
    return;
  }
  document.getElementById('load-overlay').classList.remove('open');

  setApiStatus('busy', `Loading run ${runId}...`, 10, runId);
  currentRunId = runId;
  document.getElementById('bg-cyc').textContent = currentRunId;
  document.getElementById('api-run-id').textContent = currentRunId;

  try {
    const status = await apiGet(`/api/simulation/status/${currentRunId}`);
    if (status.status === 'running' || status.status === 'pending') {
      setApiStatus('busy', 'Run still in progress — polling...', 50, currentRunId);
      await pollUntilComplete(currentRunId, (pct, st) => {
        setApiStatus('busy', `[${st.toUpperCase()}] ${pct.toFixed(0)}% complete`, pct, currentRunId);
      });
    }

    setApiStatus('busy', 'Fetching results...', 80, currentRunId);
    const [, cfgResult] = await Promise.allSettled([
      fetchAllResults(currentRunId),
      apiGet(`/api/simulation/config/${currentRunId}`).catch(() => null),
    ]);
    const runCfg = cfgResult.status === 'fulfilled' ? cfgResult.value : null;

    if (_hasConfig(runCfg)) {
      _cacheRunConfig(currentRunId, runCfg);
      const listRun = _getRunById(currentRunId);
      if (listRun) listRun.config = runCfg;
    }

    const cfgToSync = _hasConfig(runCfg)
      ? runCfg
      : (_getRunById(currentRunId)?.config || _runConfigCache.get(currentRunId) || null);
    syncControlsToConfig(cfgToSync);

    if (typeof setVital === 'function') {
      setVital('hr', '—', ''); setVital('fat', '—', ''); setVital('str', '—', ''); setVital('spo2', '—', '');
    }

    setApiStatus('', `Loaded ${currentRunId} — ${simResults.state.time.length} timesteps`, 100, currentRunId);

    document.getElementById('btn-play').disabled  = false;
    document.getElementById('btn-reset').disabled = false;

    onSimReady();
    buildTimeline();
    populateStatusPanel();
    checkHealth();

    document.getElementById('phase-disp').textContent = '◈ EXISTING RUN LOADED — PRESS PLAY';
    document.getElementById('phase-name').textContent  = 'LOADED';

  } catch (e) {
    console.error('Load failed:', e);
    setApiStatus('err', 'Load failed: ' + e.message, 0);
    alert('Failed to load run: ' + e.message);
  }
}

// ── Sync sliders to loaded run config ─────────────────────────────────────
function syncControlsToConfig(cfg) {
  if (!_hasConfig(cfg)) return;

  const mappings = [
    { ctrlId: 'ctrl-cyc',    labelId: 'lbl-cyc',    key: 'mission_duration_hours', toRaw: v => Math.round((Number(v) || 0) / 24), min: 1,  max: 365, fmt: v => `${v} days` },
    { ctrlId: 'ctrl-lam',    labelId: 'lbl-lam',    key: 'sms_rate_per_day',       toRaw: v => Math.round((Number(v) || 0) * 100), min: 1,  max: 200, fmt: v => (Number(v) / 100).toFixed(2) + '/day' },
    { ctrlId: 'ctrl-rec',    labelId: 'lbl-rec',    key: 'recovery_factor',        toRaw: v => Math.round((Number(v) || 0) * 10),  min: 5,  max: 20,  fmt: v => (Number(v) / 10).toFixed(1) + 'x' },
    { ctrlId: 'ctrl-fsens',  labelId: 'lbl-fsens',  key: 'fatigue_sensitivity',    toRaw: v => Math.round((Number(v) || 0) * 10),  min: 5,  max: 20,  fmt: v => (Number(v) / 10).toFixed(1) + 'x' },
  ];

  mappings.forEach(map => {
    if (!Object.prototype.hasOwnProperty.call(cfg, map.key)) return;
    const ctrl = document.getElementById(map.ctrlId);
    const lbl  = document.getElementById(map.labelId);
    if (!ctrl || !lbl) return;
    let raw = map.toRaw(cfg[map.key]);
    if (!Number.isFinite(raw)) return;
    raw = Math.max(map.min, Math.min(map.max, raw));
    ctrl.value = String(raw);
    lbl.textContent = map.fmt(raw);
  });

  if (Object.prototype.hasOwnProperty.call(cfg, 'baseline_sleep_quality')) {
    const ctrl = document.getElementById('ctrl-mc-slp');
    const lbl  = document.getElementById('lbl-mc-slp');
    if (ctrl && lbl) {
      const raw = Math.max(40, Math.min(100, Math.round((Number(cfg.baseline_sleep_quality) || 0.8) * 100)));
      ctrl.value = String(raw);
      lbl.textContent = (raw / 100).toFixed(2);
    }
  }
}

// ════════════════════════════════════════════════════════
// SIM TOOLTIP (legacy simple versions kept for compat)
// ════════════════════════════════════════════════════════

// Note: showSimTooltip / hideSimTooltip / scheduleHideSimTooltip /
// cancelHideSimTooltip are defined above in the tracker section.

// ════════════════════════════════════════════════════════
// RUN COMPARISON
// ════════════════════════════════════════════════════════

function renderRunComparison() {
  const section = document.getElementById('run-compare-section');
  const table   = document.getElementById('run-compare-table');
  if (!section || !table) return;

  const saved = typeof _savedRunsLoad === 'function' ? _savedRunsLoad() : [];
  if (saved.length < 2) { section.style.display = 'none'; return; }
  section.style.display = 'block';

  table.innerHTML = `
    <thead><tr>
      <th>Run ID</th><th>Status</th><th>Duration</th>
    </tr></thead>
    <tbody>
      ${saved.map(r => `
        <tr>
          <td>${r.runId || '—'}</td>
          <td>${r.status || '—'}</td>
          <td>${r.duration || '—'}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
}

function clearSavedRuns() {
  localStorage.removeItem(typeof SAVED_RUNS_KEY !== 'undefined' ? SAVED_RUNS_KEY : 'adt_saved_runs');
  if (typeof renderRunComparison === 'function') renderRunComparison();
}