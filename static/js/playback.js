// static/js/playback.js
// ════════════════════════════════════════════════════════
// PLAYBACK ENGINE
// ════════════════════════════════════════════════════════

let playState = { running: false, paused: false, idx: 0, lastTs: 0 };
let missionSecs = 0;

function startPlay() {
  if (!simResults) return;
  initCharts();
  playState = { running: true, paused: false, idx: 0, lastTs: performance.now() };
  missionSecs = 0;
  
  document.getElementById('btn-play').disabled = true;
  document.getElementById('btn-pause').disabled = false;
  
  requestAnimationFrame(playLoop);
  
  bgm.currentTime = 0;
  bgm.volume = 1;
  prevRiskLevel = 'NOMINAL';
  try { bgm.play(); } catch(e) { console.warn('BGM play blocked:', e); }
}

function togglePause() {
  markUserInteraction();
  playState.paused = !playState.paused;
  const btn = document.getElementById('btn-pause');
  btn.textContent = playState.paused ? '▶ RESUME' : '⏸ PAUSE';
  
  if (!playState.paused) {
    playState.lastTs = performance.now();
    requestAnimationFrame(playLoop);
    bgm.play();
  } else {
    bgm.pause();
  }
}

function resetPlay() {
  playState.running = false;
  playState.paused = false;
  
  initCharts();
  
  document.getElementById('btn-play').disabled = false;
  document.getElementById('btn-pause').disabled = true;
  document.getElementById('btn-pause').textContent = '⏸ PAUSE';
  document.getElementById('clock').textContent = 'T+00:00:00';
  document.getElementById('cyc-disp').textContent = 'STEP: 0 / —';
  document.getElementById('phase-name').textContent = 'RESET';
  document.getElementById('phase-disp').textContent = '◈ SIMULATION READY';
  document.getElementById('risk-badge').textContent = 'NOMINAL';
  document.getElementById('risk-badge').className = '';
  
  setVital('hr', '—', '');
  setVital('fat', '—', '');
  setVital('str', '—', '');
  setVital('spo2', '—', '');
  
  _simState = null;
  prevRiskLevel = 'NOMINAL';
  highRiskLighting = false;
  hideInspectTip();
  
  bgm.pause();
  bgm.currentTime = 0;
  bgm.volume = 1;
  
  if (simResults) buildTimeline();
}

function getPhaseFromTime(timeH) {
  const h = timeH % 24;
  if (h < 8) return 'WORK';
  if (h < 10) return 'EVENT';
  if (h < 16) return 'ACTIVE';
  return 'SLEEP';
}

function playLoop(ts) {
  if (!playState.running || playState.paused) return;
  
  const speed = +document.getElementById('ctrl-spd').value / 3;
  playState.lastTs = ts;
  
  const stepsPerSec = Math.max(0.01, (speed * 4.8) / 60);
  playState.idx = Math.min(playState.idx + stepsPerSec, simResults.state.time.length - 1);
  
  const state = simResults.state;
  const idx = Math.floor(playState.idx);
  const total = state.time.length;
  const timeH = state.time[idx] / 60;
  
  const ms = Math.floor(timeH * 3600);
  document.getElementById('clock').textContent =
    `T+${String(Math.floor(ms / 3600)).padStart(2, '0')}:${String(Math.floor((ms % 3600) / 60)).padStart(2, '0')}:${String(ms % 60).padStart(2, '0')}`;
  
  const fat = state.fatigue[idx];
  const hr = Math.round(state.hr[idx]);
  const slp = state.sleep_quality[idx];
  const motRaw = Array.isArray(state.motion_severity) ? (state.motion_severity[idx] || 0) : 0;
  const mot01 = Math.min(1, motRaw / 5.0);
  const fat01 = Math.min(1, state.fatigue[idx] / 10.0);
  const dayHour = (state.time[idx] / 60.0) % 24;
  const circ = 0.08 + 0.07 * Math.sin(2 * Math.PI * (dayHour - 4) / 24);
  const backendStr = Array.isArray(state.stress) && state.stress[idx] > 0.01 ? state.stress[idx] : null;
  
  const derivedStr = Math.min(0.95,
    0.12 + circ + fat01 * 0.42 + mot01 * 0.55
  );
  
  const str = backendStr !== null ? Math.max(backendStr, derivedStr * 0.7) : derivedStr;
  const spo2 = Math.max(93.0, 98.8 - str * 2.4 - fat01 * 1.2);
  const phase = getPhaseFromTime(timeH);
  const rsk = Math.min(1, (fat / 10) * 0.5 + (str || 0) * 0.3 + Math.max(0, (1 - slp)) * 0.2);
  
  const riskLevel = getRiskTier(rsk);
  playRiskTransition(prevRiskLevel, riskLevel);
  prevRiskLevel = riskLevel;
  
  _simState = {
    fatigue: fat / 10,
    fatigueIndex: fat,
    hr,
    smsSev: mot01,
    motionSeverity: motRaw,
    stress: str || 0,
    risk: rsk,
    phase,
    spo2,
  };
  
  document.getElementById('cyc-disp').textContent = `STEP: ${idx + 1} / ${total}`;
  document.getElementById('phase-name').textContent = phase;
  
  const phaseLabels = {
    'WORK': '◈ WORK PHASE — ASTRONAUT ACTIVE',
    'EVENT': '◈ EVENT WINDOW',
    'ACTIVE': '◈ ACTIVE PHASE',
    'SLEEP': '◈ SLEEP & RECOVERY PHASE',
  };
  document.getElementById('phase-disp').textContent = phaseLabels[phase] || '';
  
  const fatClass = fat > 7 ? 'R' : fat > 5 ? 'W' : '';
  const hrClass = hr > 100 ? 'W' : hr > 115 ? 'R' : '';
  const spo2C = spo2 < 94 ? 'R' : spo2 < 97 ? 'W' : '';
  
  setVital('hr', hr, hrClass);
  setVital('fat', fat.toFixed(2), fatClass);
  setVital('str', (str || 0).toFixed(2), (str || 0) > .6 ? 'W' : '');
  setVital('spo2', (+spo2).toFixed(1), spo2C);
  
  const rb = document.getElementById('risk-badge');
  if (rsk >= .7) { rb.textContent = 'CRITICAL'; rb.className = 'R'; }
  else if (rsk >= .4) { rb.textContent = 'ELEVATED'; rb.className = 'Y'; }
  else { rb.textContent = 'NOMINAL'; rb.className = ''; }
  
  if (idx % 4 === 0) {
    for (const key of ['fat', 'hr', 'slp', 'rsk']) {
      if (!chartScrubbing[key]) {
        const dataKey = { fat: 'fat', hr: 'hr', slp: 'slp', rsk: 'risk' }[key];
        const ds = charts[key].data;
        const start = Math.max(0, idx - 29);
        ds.labels = fullChartData ? fullChartData.time.slice(start, idx + 1) : [];
        ds.datasets[0].data = fullChartData ? fullChartData[dataKey].slice(start, idx + 1) : [];
        charts[key].update('none');
        document.getElementById('sl-' + key).value = idx;
        document.getElementById('st-' + key).textContent = fullChartData ? fullChartData.time[idx] : `${timeH.toFixed(0)}h`;
      }
    }
  }
  
  tlActivate(Math.floor(timeH / 24));
  
  if (playState.idx >= total - 1) {
    endMission();
    return;
  }
  
  requestAnimationFrame(playLoop);
}

function setVital(id, val, cls) {
  const el = document.getElementById('v-' + id);
  el.textContent = val;
  el.className = 'vval' + (cls ? ' ' + cls : '');
}

function endMission() {
  playState.running = false;
  chartScrubbing = { fat: false, hr: false, slp: false, rsk: false };
  
  for (const key of ['fat', 'hr', 'slp', 'rsk']) {
    const lb = document.getElementById('lv-' + key);
    if (lb) lb.classList.remove('visible');
  }
  
  fadeOutBGM(3000);
  document.getElementById('phase-disp').textContent = '◈ MISSION COMPLETE — GENERATING REPORT';
  document.getElementById('risk-badge').className = '';
  document.getElementById('risk-badge').textContent = 'COMPLETE';
  
  onPlaybackComplete();
  showMC();
  showReport();
  
  if (currentRunId) loadAnalytics(currentRunId);
}