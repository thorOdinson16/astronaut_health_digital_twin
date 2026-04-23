// static/js/utils.js
// ════════════════════════════════════════════════════════
// SHARED HELPERS & CONFIG
// ════════════════════════════════════════════════════════

const API_BASE = 'http://127.0.0.1:8000';

// Parameter labels & groups
const _PARAM_LABELS = {
  mission_duration_hours: 'Mission Duration',
  time_step_minutes: 'Time Step',
  num_astronauts: 'Crew Size',
  astronaut_id: 'Astronaut ID',
  sms_rate_per_day: 'SMS Rate',
  recovery_factor: 'Recovery Factor',
  fatigue_sensitivity: 'Fatigue Sensitivity',
  baseline_hr: 'Baseline HR',
  baseline_sleep_quality: 'Baseline Sleep',
  initial_fatigue: 'Initial Fatigue',
  enable_motion_sickness: 'Motion Sickness',
  enable_sleep_disruption: 'Sleep Disruption',
  use_biogears: 'BioGears',
  save_trajectories: 'Save Trajectories',
  save_events: 'Save Events',
  biogears_scenario_path: 'BioGears Scenario',
};

const _PARAM_GROUPS = [
  { title: 'MISSION', keys: ['mission_duration_hours', 'time_step_minutes', 'num_astronauts', 'astronaut_id'] },
  { title: 'DYNAMICS', keys: ['sms_rate_per_day', 'recovery_factor', 'fatigue_sensitivity', 'baseline_hr', 'baseline_sleep_quality', 'initial_fatigue'] },
  { title: 'FEATURE FLAGS', keys: ['enable_motion_sickness', 'enable_sleep_disruption', 'use_biogears'] },
  { title: 'OUTPUT', keys: ['save_trajectories', 'save_events', 'biogears_scenario_path'] },
];

const PRESET_DEFS = {
  custom: { label: 'CUSTOM', desc: 'CUSTOM MISSION PROFILE' },
  iss: { label: 'ISS EXPEDITION', desc: 'LOW-INTENSITY LONG-DURATION LEO CREW RHYTHM', durationDays: 30, lambdaPerDay: 0.05, mcSleep: 0.80, mcFatThr: 5.0 },
  mars: { label: 'MARS TRANSIT', desc: 'DEEP-SPACE TRANSIT WITH ELEVATED CUMULATIVE LOAD', durationDays: 180, lambdaPerDay: 0.12, mcSleep: 0.65, mcFatThr: 6.5 },
  eva: { label: 'EVA PREPARATION', desc: 'SHORT MISSION WITH DENSE WORK CYCLE AND PRE-BURNOUT RISK', durationDays: 7, lambdaPerDay: 0.08, mcSleep: 0.70, mcFatThr: 4.5 },
  lunar: { label: 'LUNAR SURFACE', desc: 'SURFACE OPERATIONS WITH SUSTAINED MODERATE PHYSIOLOGICAL STRAIN', durationDays: 14, lambdaPerDay: 0.09, mcSleep: 0.72, mcFatThr: 5.5 },
};

const SAVED_RUNS_KEY = 'adt_saved_runs';
const TOURED_KEY = 'adt_toured';

// Utility functions
function _escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function _isPlainObject(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function _hasConfig(config) {
  return _isPlainObject(config) && Object.keys(config).length > 0;
}

function _toNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function _compactNumber(value, digits = 2) {
  const n = _toNumber(value);
  if (n === null) return '—';
  return n.toFixed(digits).replace(/\.0+$/, '').replace(/(\.\d*[1-9])0+$/, '$1');
}

function _parseLikeDate(value) {
  if (value === null || value === undefined || value === '') return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function _truncateText(value, maxLen = 56) {
  const text = String(value);
  return text.length > maxLen ? text.slice(0, maxLen - 1) + '…' : text;
}

function _paramLabel(key) {
  return _PARAM_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function _formatParamValue(key, value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'boolean') return value ? 'ON' : 'OFF';
  
  const n = _toNumber(value);
  switch (key) {
    case 'mission_duration_hours': {
      if (n === null) return String(value);
      const days = n / 24;
      return `${_compactNumber(days, days % 1 === 0 ? 0 : 1)} days (${_compactNumber(n, 0)} h)`;
    }
    case 'time_step_minutes': return n === null ? String(value) : `${_compactNumber(n, 1)} min`;
    case 'sms_rate_per_day': return n === null ? String(value) : `${_compactNumber(n, 2)} /day`;
    case 'recovery_factor': case 'fatigue_sensitivity': return n === null ? String(value) : `${_compactNumber(n, 2)}x`;
    case 'baseline_hr': return n === null ? String(value) : `${_compactNumber(n, 0)} bpm`;
    case 'baseline_sleep_quality': return n === null ? String(value) : _compactNumber(n, 2);
    case 'initial_fatigue': return n === null ? String(value) : _compactNumber(n, 2);
    default: break;
  }
  
  if (Array.isArray(value)) return _truncateText(value.join(', '));
  if (_isPlainObject(value)) return _truncateText(JSON.stringify(value));
  return _truncateText(String(value));
}

function _buildParamSectionHtml(title, keys, config) {
  const rows = keys
    .filter(key => Object.prototype.hasOwnProperty.call(config, key))
    .map(key => {
      const label = _escapeHtml(_paramLabel(key));
      const val = _escapeHtml(_formatParamValue(key, config[key]));
      return `<div class="param-row"><span class="param-key">${label}</span><span class="param-val" title="${val}">${val}</span></div>`;
    });
  
  if (!rows.length) return '';
  return `<div class="param-section"><div class="param-section-title">${_escapeHtml(title)}</div>${rows.join('')}</div>`;
}

function _renderParamHtml(config, emptyText) {
  if (!_hasConfig(config)) {
    return `<div class="param-empty">${_escapeHtml(emptyText || 'PARAMETERS UNAVAILABLE')}</div>`;
  }
  
  const used = new Set();
  let html = _PARAM_GROUPS
    .map(group => {
      group.keys.forEach(k => used.add(k));
      return _buildParamSectionHtml(group.title, group.keys, config);
    })
    .filter(Boolean)
    .join('');
  
  const extraKeys = Object.keys(config).filter(k => !used.has(k));
  if (extraKeys.length) {
    html += _buildParamSectionHtml('OTHER', extraKeys, config);
  }
  return html || `<div class="param-empty">${_escapeHtml(emptyText || 'PARAMETERS UNAVAILABLE')}</div>`;
}

function _setParamPanelContent(contentId, config, emptyText) {
  const content = document.getElementById(contentId);
  if (!content) return;
  content.innerHTML = _renderParamHtml(config, emptyText);
}

function getRiskTier(risk) {
  if (risk >= 0.85) return 'CRITICAL';
  if (risk >= 0.7) return 'HIGH';
  if (risk >= 0.4) return 'MODERATE';
  return 'NOMINAL';
}

function _savedRunsLoad() {
  try {
    const raw = localStorage.getItem(SAVED_RUNS_KEY);
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr : [];
  } catch { return []; }
}

function _savedRunsStore(arr) {
  localStorage.setItem(SAVED_RUNS_KEY, JSON.stringify(arr.slice(0, 4)));
}