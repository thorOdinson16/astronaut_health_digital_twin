// static/js/charts.js
// ════════════════════════════════════════════════════════
// CHART MANAGEMENT
// ════════════════════════════════════════════════════════

let charts = {};
const CYAN = '#00d4ff', AMBER = '#ffaa00', GREEN = '#00e87a', RED = '#ff4040';
let fullChartData = null;
let chartScrubbing = { fat: false, hr: false, slp: false, rsk: false };
let fullChartInstance = null;

const baseOpts = (yMin, yMax, color) => ({
  type: 'line',
  data: {
    labels: [],
    datasets: [{
      data: [],
      borderColor: color,
      borderWidth: 1.5,
      pointRadius: 1.5,
      pointBackgroundColor: color,
      tension: .4,
      fill: true,
      backgroundColor: color + '18'
    }]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 200 },
    plugins: { legend: { display: false } },
    scales: {
      x: {
        ticks: { color: '#7aa8c8', font: { size: 9, family: 'Courier New' }, maxTicksLimit: 8 },
        grid: { color: '#152844', lineWidth: .5 },
        border: { color: '#152844' }
      },
      y: {
        min: yMin,
        max: yMax,
        ticks: { color: '#7aa8c8', font: { size: 9, family: 'Courier New' }, maxTicksLimit: 5 },
        grid: { color: '#152844', lineWidth: .5 },
        border: { color: '#152844' }
      }
    }
  }
});

function initCharts() {
  if (charts.fat) { Object.values(charts).forEach(c => c.destroy()); charts = {}; }
  charts.fat = new Chart(document.getElementById('ch-fat'), baseOpts(0, 10, CYAN));
  charts.hr = new Chart(document.getElementById('ch-hr'), baseOpts(40, 160, AMBER));
  charts.slp = new Chart(document.getElementById('ch-slp'), baseOpts(0, 1, GREEN));
  charts.rsk = new Chart(document.getElementById('ch-rsk'), baseOpts(0, 1, RED));
}

function computeFullChartData() {
  if (!simResults) return;
  const state = simResults.state;
  const total = state.time.length;
  
  const timeLabels = [], fatArr = [], hrArr = [], slpArr = [], riskArr = [];
  
  for (let i = 0; i < total; i++) {
    const timeH = state.time[i] / 60;
    timeLabels.push(`${timeH.toFixed(0)}h`);
    
    const fat = state.fatigue[i];
    const hr = Math.round(state.hr[i]);
    const slp = state.sleep_quality[i];
    
    const motRaw = Array.isArray(state.motion_severity) ? (state.motion_severity[i] || 0) : 0;
    const mot01 = Math.min(1, motRaw / 5.0);
    const fat01 = Math.min(1, fat / 10.0);
    const dayHour = (state.time[i] / 60.0) % 24;
    const circ = 0.08 + 0.07 * Math.sin(2 * Math.PI * (dayHour - 4) / 24);
    const derivedStr = Math.min(0.95, 0.12 + circ + fat01 * 0.42 + mot01 * 0.55);
    const rsk = Math.min(1, (fat / 10) * 0.5 + (derivedStr || 0) * 0.3 + Math.max(0, (1 - slp)) * 0.2);
    
    fatArr.push(+fat.toFixed(2));
    hrArr.push(hr);
    slpArr.push(+slp.toFixed(3));
    riskArr.push(+rsk.toFixed(3));
  }
  
  fullChartData = { time: timeLabels, fat: fatArr, hr: hrArr, slp: slpArr, risk: riskArr };
  
  for (const key of ['fat', 'hr', 'slp', 'rsk']) {
    const row = document.getElementById('scr-' + key);
    if (row) {
      row.classList.add('visible');
      document.getElementById('sl-' + key).max = total - 1;
    }
  }
}

function chartScrub(key, idx) {
  if (!fullChartData) return;
  chartScrubbing[key] = true;
  const total = fullChartData.time.length;
  idx = Math.max(0, Math.min(idx, total - 1));
  
  const dataKey = { fat: 'fat', hr: 'hr', slp: 'slp', rsk: 'risk' }[key];
  
  const liveBtn = document.getElementById('lv-' + key);
  if (liveBtn) liveBtn.classList.add('visible');
  
  document.getElementById('sl-' + key).value = idx;
  
  const start = Math.max(0, idx - 29);
  const ds = charts[key].data;
  ds.labels = fullChartData.time.slice(start, idx + 1);
  ds.datasets[0].data = fullChartData[dataKey].slice(start, idx + 1);
  charts[key].update('none');
  
  document.getElementById('st-' + key).textContent = fullChartData.time[idx];
}

function chartPrev(key) {
  if (!fullChartData) return;
  const idx = +document.getElementById('sl-' + key).value;
  chartScrub(key, idx - 1);
}

function chartNext(key) {
  if (!fullChartData) return;
  const idx = +document.getElementById('sl-' + key).value;
  chartScrub(key, idx + 1);
}

function chartResume(key) {
  chartScrubbing[key] = false;
  const liveBtn = document.getElementById('lv-' + key);
  if (liveBtn) liveBtn.classList.remove('visible');
}

function openFullChart(key, title) {
  if (!fullChartData) return;
  const overlay = document.getElementById('full-chart-overlay');
  overlay.classList.add('open');
  document.getElementById('full-chart-title').textContent = title + ' — FULL TIMELINE';
  
  if (fullChartInstance) { fullChartInstance.destroy(); fullChartInstance = null; }
  
  const colorMap = { fat: CYAN, hr: AMBER, slp: GREEN, rsk: RED };
  const yRangeMap = { fat: [0, 10], hr: [40, 160], slp: [0, 1], rsk: [0, 1] };
  const dataKeyMap = { fat: 'fat', hr: 'hr', slp: 'slp', rsk: 'risk' };
  
  const color = colorMap[key];
  const [yMin, yMax] = yRangeMap[key];
  const dataKey = dataKeyMap[key];
  
  fullChartInstance = new Chart(document.getElementById('ch-full'), {
    type: 'line',
    data: {
      labels: fullChartData.time,
      datasets: [{
        data: fullChartData[dataKey],
        borderColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        fill: true,
        backgroundColor: color + '18'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#7aa8c8', font: { size: 9, family: 'Courier New' }, maxTicksLimit: 15 },
          grid: { color: '#152844', lineWidth: 0.5 },
          border: { color: '#152844' }
        },
        y: {
          min: yMin,
          max: yMax,
          ticks: { color: '#7aa8c8', font: { size: 9, family: 'Courier New' }, maxTicksLimit: 6 },
          grid: { color: '#152844', lineWidth: 0.5 },
          border: { color: '#152844' }
        }
      }
    }
  });
}

function closeFullChart(e) {
  if (e && e.target !== document.getElementById('full-chart-overlay')) return;
  document.getElementById('full-chart-overlay').classList.remove('open');
  if (fullChartInstance) { fullChartInstance.destroy(); fullChartInstance = null; }
}

function buildTimeline() {
  if (!simResults) return;
  const state = simResults.state;
  const events = simEvents;
  const times = state.time;
  const fat = state.fatigue;
  const totalH = times[times.length - 1] / 60;
  const timesH = times.map(t => t / 60);
  const nDays = Math.ceil(totalH / 24);
  
  document.getElementById('tl-meta').textContent = `${nDays} days · ${times.length} timesteps · ${events.length} events`;
  
  const track = document.getElementById('tl-track');
  track.innerHTML = '';
  
  for (let d = 0; d < nDays; d++) {
    const dStart = d * 24, dEnd = (d + 1) * 24;
    const indices = timesH.map((t, i) => t >= dStart && t < dEnd ? i : -1).filter(i => i >= 0);
    const avgFat = indices.length ? indices.reduce((s, i) => s + fat[i], 0) / indices.length : 0;
    
    const dayEvents = events.filter(e => {
      const t = e.onset_time || e.simulation_time || e.start_time || 0;
      return t >= dStart && t < dEnd;
    });
    
    const hasSms = dayEvents.some(e => e.type && e.type.toLowerCase().includes('motion'));
    const hasDis = dayEvents.some(e => e.type && e.type.toLowerCase().includes('sleep'));
    
    const blk = document.createElement('div');
    blk.className = 'dayblk';
    blk.id = `tlblk-${d}`;
    
    const pills = [
      '<span class="pill pw">WORK</span>',
      hasSms ? '<span class="pill ps">SMS</span>' : '',
      hasDis ? '<span class="pill pd">DISRPT</span>' : '',
      '<span class="pill psl">SLEEP</span>',
    ].join('');
    
    const fatW = Math.min(100, (avgFat / 10 * 100)).toFixed(0);
    const fatC = avgFat < 3 ? '#00e87a' : avgFat < 6 ? '#ffaa00' : '#ff4040';
    
    blk.innerHTML = `<div class="dlabel">DAY ${d + 1}</div><div class="dpills">${pills}</div><div class="fbar"><div class="ffill" style="width:${fatW}%;background:${fatC}"></div></div>`;
    track.appendChild(blk);
  }
}

function tlActivate(idx) {
  document.querySelectorAll('.dayblk').forEach((b, i) => {
    b.className = 'dayblk' + (i === idx ? ' active' : i < idx ? ' done' : '');
  });
  
  const track = document.getElementById('tl-track');
  const active = document.getElementById('tlblk-' + idx);
  if (active && track) {
    active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
  }
}