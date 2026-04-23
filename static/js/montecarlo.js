// static/js/montecarlo.js
// ════════════════════════════════════════════════════════
// MONTE CARLO SIMULATION
// ════════════════════════════════════════════════════════

let mcFatigueChart = null;
let mcSleepChart = null;
let _mcProgTimer = null;

function _mcProgStart(nRuns) {
  const wrap = document.getElementById('mc-progress-wrap');
  const results = document.getElementById('mc-results-wrap');
  if (wrap) wrap.style.display = 'block';
  if (results) results.style.display = 'none';
  
  const bar = document.getElementById('mc-prog-bar');
  const pct = document.getElementById('mc-prog-pct');
  const runsEl = document.getElementById('mc-prog-runs');
  const label = document.getElementById('mc-prog-label');
  const elEl = document.getElementById('mcs-elapsed-val');
  const etaEl = document.getElementById('mcs-eta-val');
  const trajEl = document.getElementById('mcs-traj-val');
  
  const start = Date.now();
  let progress = 0;
  const estMs = Math.min(nRuns * 25 + 400, 8000);
  
  _mcProgTimer = setInterval(() => {
    const elapsed = (Date.now() - start) / 1000;
    progress = Math.min(0.90, 1 - Math.exp(-elapsed / (estMs / 1000) * 2.5));
    const fakeDone = Math.round(progress * nRuns);
    
    if (bar) bar.style.width = (progress * 100).toFixed(1) + '%';
    if (pct) pct.textContent = (progress * 100).toFixed(0) + '%';
    if (runsEl) runsEl.textContent = `${fakeDone} / ${nRuns} trajectories`;
    
    if (label) {
      if (progress < 0.2) label.textContent = 'SEEDING RANDOM TRAJECTORIES...';
      else if (progress < 0.5) label.textContent = 'SIMULATING PHYSIOLOGICAL RESPONSES...';
      else if (progress < 0.75) label.textContent = 'COMPUTING FATIGUE ENVELOPES...';
      else label.textContent = 'AGGREGATING RISK DISTRIBUTIONS...';
    }
    
    if (trajEl) trajEl.textContent = fakeDone;
    if (elEl) elEl.textContent = elapsed.toFixed(1) + 's';
    const remaining = elapsed > 0 ? (elapsed / progress) - elapsed : '—';
    if (etaEl) etaEl.textContent = typeof remaining === 'number' ? remaining.toFixed(1) + 's' : '—';
  }, 80);
}

function _mcProgFinish() {
  clearInterval(_mcProgTimer);
  _mcProgTimer = null;
  
  const bar = document.getElementById('mc-prog-bar');
  const pct = document.getElementById('mc-prog-pct');
  const runs = document.getElementById('mc-prog-runs');
  const lbl = document.getElementById('mc-prog-label');
  
  if (bar) bar.style.width = '100%';
  if (pct) pct.textContent = '100%';
  if (lbl) lbl.textContent = 'COMPLETE — RENDERING ENVELOPES';
  
  setTimeout(() => {
    const wrap = document.getElementById('mc-progress-wrap');
    const results = document.getElementById('mc-results-wrap');
    if (wrap) wrap.style.display = 'none';
    if (results) results.style.display = 'block';
  }, 500);
}

function _mcProgError(msg) {
  clearInterval(_mcProgTimer);
  _mcProgTimer = null;
  
  const lbl = document.getElementById('mc-prog-label');
  const bar = document.getElementById('mc-prog-bar');
  if (lbl) { lbl.textContent = '✕ ' + msg; lbl.style.color = 'var(--red)'; }
  if (bar) bar.style.background = 'var(--red)';
}

async function runMonteCarlo() {
  const btn = document.getElementById('btn-mc');
  if (btn) { btn.disabled = true; btn.textContent = '⟳ RUNNING MC...'; }
  
  document.getElementById('tl-section').style.display = 'none';
  document.getElementById('mission-risk-section').style.display = 'none';
  document.getElementById('mc-section').style.display = 'block';
  
  if (mcFatigueChart) { mcFatigueChart.destroy(); mcFatigueChart = null; }
  if (mcSleepChart) { mcSleepChart.destroy(); mcSleepChart = null; }
  
  document.getElementById('mc-stats').innerHTML = '';
  const concEl = document.getElementById('mc-conclusions');
  if (concEl) { concEl.style.display = 'none'; concEl.innerHTML = ''; }
  
  const nRuns = parseInt(document.getElementById('ctrl-mc-runs')?.value || 50);
  _mcProgStart(nRuns);
  
  try {
    const cfg = {
      n_runs: nRuns,
      mission_duration_hours: parseInt(document.getElementById('ctrl-cyc')?.value || 30) * 24,
      time_step_minutes: 30,
      ms_lambda: parseFloat(document.getElementById('ctrl-lam')?.value || 72) / 2400,
      baseline_sleep_quality: parseFloat(document.getElementById('ctrl-mc-slp')?.value || 80) / 100,
      risk_fatigue_threshold: parseFloat(document.getElementById('ctrl-mc-fat-thr')?.value || 50) / 10,
      risk_sleep_threshold: parseFloat(document.getElementById('ctrl-mc-slp-thr')?.value || 40) / 100,
    };
    
    const res = await fetch(`${API_BASE}/api/simulation/monte-carlo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    
    if (!res.ok) throw new Error(`MC endpoint returned ${res.status}`);
    const mc = await res.json();
    
    _mcProgFinish();
    setTimeout(() => {
      renderMCCharts(mc);
      renderMCStats(mc);
      renderMCConclusions(mc);
    }, 520);
    
  } catch (e) {
    console.error('MC run failed:', e);
    _mcProgError(e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '◈ RUN MONTE CARLO'; }
  }
}

function renderMCCharts(mc) {
  const env = mc.envelopes;
  const t = env.time_hours;
  
  const fatCtx = document.getElementById('ch-mc-t')?.getContext('2d');
  if (fatCtx) {
    if (mcFatigueChart) mcFatigueChart.destroy();
    mcFatigueChart = new Chart(fatCtx, {
      type: 'line',
      data: {
        labels: t.map(h => `${Math.round(h / 24)}d`),
        datasets: [
          { label: 'Mean fatigue', data: env.fatigue_mean, borderColor: '#ffaa00', borderWidth: 1.5, pointRadius: 0, fill: false },
          { label: 'Max', data: env.fatigue_max, borderColor: 'rgba(255,64,64,0.4)', borderWidth: 1, pointRadius: 0 },
          { label: 'Min', data: env.fatigue_min, borderColor: 'rgba(255,64,64,0.2)', borderWidth: 1, pointRadius: 0 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#7aa8c8', font: { size: 8 } }, grid: { color: '#152844' } },
          y: { ticks: { color: '#7aa8c8', font: { size: 8 } }, grid: { color: '#152844' }, min: 0, max: 10 }
        }
      }
    });
  }
  
  const slpCtx = document.getElementById('ch-mc-h')?.getContext('2d');
  if (slpCtx) {
    if (mcSleepChart) mcSleepChart.destroy();
    mcSleepChart = new Chart(slpCtx, {
      type: 'line',
      data: {
        labels: t.map(h => `${Math.round(h / 24)}d`),
        datasets: [
          { label: 'Mean sleep Q', data: env.sleep_mean, borderColor: '#aa88ff', borderWidth: 1.5, pointRadius: 0, fill: false },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#7aa8c8', font: { size: 8 } }, grid: { color: '#152844' } },
          y: { ticks: { color: '#7aa8c8', font: { size: 8 } }, grid: { color: '#152844' }, min: 0, max: 1 }
        }
      }
    });
  }
}

function renderMCStats(mc) {
  const rs = mc.risk_summary || {};
  const el = document.getElementById('mc-stats');
  if (!el) return;
  
  el.innerHTML = `
    <div class="mcstat"><div class="mcslabel">n RUNS</div><div class="mcsval">${mc.n_runs}</div></div>
    <div class="mcstat"><div class="mcslabel">P(FAT RISK) MEAN</div><div class="mcsval W">${((rs.mean_prob_fatigue_risk || 0) * 100).toFixed(1)}%</div></div>
    <div class="mcstat"><div class="mcslabel">P95 PEAK FAT</div><div class="mcsval R">${(rs.p95_peak_fatigue || 0).toFixed(2)}</div></div>
    <div class="mcstat"><div class="mcslabel">MED RECOVERY</div><div class="mcsval">${(rs.median_recovery_hours || 0).toFixed(1)}h</div></div>
  `;
}

function renderMCConclusions(mc) {
  const el = document.getElementById('mc-conclusions');
  if (!el || !mc.conclusions) return;
  el.style.display = 'block';
  el.innerHTML = mc.conclusions.map(c =>
    `<p style="margin-bottom:6px;line-height:1.5">${c}</p>`
  ).join('');
}

async function loadAnalytics(runId) {
  try {
    const res = await fetch(`${API_BASE}/api/data/analytics/${runId}`);
    if (!res.ok) return;
    const data = await res.json();
    currentAnalytics = data;
    
    renderRiskWindows(data.risk_report);
    renderTrends(data.trend_report);
    renderCumulativeLoad(data.risk_report);
    
    const actions = document.getElementById('risk-actions');
    if (actions && !playState.running) actions.style.display = 'flex';
  } catch (e) {
    console.warn('Analytics fetch failed:', e);
  }
}

function renderRiskWindows(riskReport) {
  if (!riskReport) return;
  const stats = document.getElementById('mission-risk-stats');
  const th = riskReport.threshold_metrics || {};
  const fat = th.fatigue || {};
  const slp = th.sleep_quality || {};
  
  const level = riskReport.overall_risk_level || 'UNKNOWN';
  const colour = level === 'CRITICAL' ? 'var(--red)' : level === 'HIGH' ? 'var(--amber)' : level === 'MODERATE' ? 'var(--cyan)' : 'var(--green)';
  
  stats.innerHTML = `
    <div class="mcstat"><div class="mcslabel">RISK LEVEL</div><div class="mcsval" style="color:${colour}">${level}</div></div>
    <div class="mcstat"><div class="mcslabel">P(CRIT SLEEP)</div><div class="mcsval ${slp.prob_critical > 0.3 ? 'W' : ''}">${((slp.prob_critical || 0) * 100).toFixed(1)}%</div></div>
    <div class="mcstat"><div class="mcslabel">P(POOR SLEEP)</div><div class="mcsval ${slp.prob_poor > 0.3 ? 'W' : ''}">${((slp.prob_poor || 0) * 100).toFixed(1)}%</div></div>
    <div class="mcstat"><div class="mcslabel">AT-RISK WINDOWS</div><div class="mcsval R">${riskReport.n_risk_windows || 0}</div></div>
  `;
}

function renderTrends(trendReport) {
  if (!trendReport || !trendReport.trends) return;
  const trends = trendReport.trends;
  const el = document.getElementById('mission-risk-conclusions');
  if (!el) return;
  
  const lines = Object.entries(trends)
    .filter(([, t]) => t.significant)
    .map(([var_, t]) =>
      `<b>${var_.replace('_', ' ')}</b>: ${t.direction} (${(t.slope_per_day > 0 ? '+' : '')}${t.slope_per_day.toFixed(3)}/day, R²=${t.r_squared.toFixed(2)})`
    );
  
  if (lines.length === 0) {
    el.style.display = 'none';
    return;
  }
  el.style.display = 'block';
  el.innerHTML = '<b style="color:var(--cyan);letter-spacing:2px">TREND ANALYSIS</b><br>' + lines.join('<br>');
}

function renderCumulativeLoad(riskReport) {
  if (!riskReport || !riskReport.cumulative_load) return;
  const cl = riskReport.cumulative_load;
  const el = document.getElementById('mission-risk-conclusions');
  if (!el) return;
  
  const loadHtml =
    `<br><b style="color:var(--cyan);letter-spacing:2px">CUMULATIVE LOAD</b><br>` +
    `Fatigue load: ${Number(cl.cumulative_fatigue_load || 0).toFixed(1)} F·hr &nbsp;|&nbsp; ` +
    `Sleep debt: ${Number(cl.cumulative_sleep_debt || 0).toFixed(1)} hr &nbsp;|&nbsp; ` +
    `Avg fatigue/day: ${Number(cl.fatigue_integral_per_day || 0).toFixed(2)}`;
  
  el.style.display = 'block';
  el.innerHTML += loadHtml;
}