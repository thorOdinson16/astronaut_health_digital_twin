// static/js/export.js
// ════════════════════════════════════════════════════════
// PDF EXPORT — exportSimReport() + exportMCReport()
// ════════════════════════════════════════════════════════

// ── Shared PDF helpers factory ────────────────────────────────────────────
function _buildPdfHelpers(pdf, PW, PH, M, CW, pageNumRef) {
  const C = {
    pageBg:   [255, 255, 255],
    hdrDark:  [8, 18, 36],
    hdrCyan:  [0, 195, 235],
    accent:   [15, 75, 150],
    accentLt: [232, 242, 255],
    text:     [25, 30, 38],
    muted:    [105, 120, 140],
    border:   [205, 215, 228],
    rowAlt:   [246, 250, 254],
    red:      [190, 35, 35],
    amber:    [170, 100, 0],
    green:    [0, 125, 60],
    orange:   [190, 90, 0],
  };
  const riskColor = l =>
    l === 'CRITICAL' ? C.red :
    l === 'HIGH'     ? C.amber :
    l === 'MODERATE' ? C.orange : C.green;

  function footer(reportTitle) {
    pdf.setDrawColor(...C.border);
    pdf.line(M, PH - 12, PW - M, PH - 12);
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7); pdf.setTextColor(...C.muted);
    pdf.text('ASTRONAUT DIGITAL TWIN  ·  ' + (reportTitle || 'REPORT'), M, PH - 7);
    pdf.text(`Page ${pageNumRef.n}`, PW - M, PH - 7, { align: 'right' });
  }

  function newPage(reportTitle) {
    footer(reportTitle);
    pdf.addPage(); pageNumRef.n++;
    pdf.setFillColor(...C.pageBg); pdf.rect(0, 0, PW, PH, 'F');
    pdf.setFillColor(...C.accent); pdf.rect(0, 0, PW, 1.5, 'F');
    return 14;
  }

  function section(title, yRef) {
    let y = yRef.v;
    if (y + 14 > PH - 18) { y = newPage(); yRef.v = y; }
    y += 3;
    pdf.setFillColor(...C.accent); pdf.rect(M, y, CW, 7.5, 'F');
    pdf.setFillColor(...C.hdrCyan); pdf.rect(M, y, 3, 7.5, 'F');
    pdf.setFont('helvetica', 'bold'); pdf.setFontSize(8.5); pdf.setTextColor(255, 255, 255);
    pdf.text(title, M + 6, y + 5.2);
    y += 11; yRef.v = y;
  }

  function kvRow(label, value, valueColor, pillText, pillColor, yRef) {
    let y = yRef.v;
    if (y + 7 > PH - 18) { y = newPage(); yRef.v = y; }
    pdf.setFillColor(...C.pageBg); pdf.rect(M, y - 4, CW, 6.5, 'F');
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(8); pdf.setTextColor(...C.muted);
    pdf.text(label, M + 2, y);
    pdf.setFont('helvetica', 'bold'); pdf.setTextColor(...(valueColor || C.text));
    pdf.text(String(value ?? '—'), M + 75, y);
    if (pillText) {
      const pw = 28, px = PW - M - 2;
      pdf.setFillColor(...(pillColor || C.green));
      pdf.roundedRect(px - pw, y - 4.2, pw, 5.5, 1, 1, 'F');
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(6.5); pdf.setTextColor(255, 255, 255);
      pdf.text(pillText, px - pw / 2, y - 0.5, { align: 'center' });
    }
    y += 6.5; yRef.v = y;
  }

  function divider(yRef) {
    pdf.setDrawColor(...C.border); pdf.setLineWidth(0.3);
    pdf.line(M, yRef.v, PW - M, yRef.v);
    yRef.v += 4;
  }

  function para(text, indent, yRef) {
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(8.2); pdf.setTextColor(...C.text);
    const lines = pdf.splitTextToSize(text, CW - (indent || 0) - 2);
    lines.forEach(l => {
      if (yRef.v + 5 > PH - 18) { yRef.v = newPage(); }
      pdf.text(l, M + (indent || 0), yRef.v);
      yRef.v += 4.6;
    });
    yRef.v += 1;
  }

  function tbl(rows, colW, yRef) {
    const RH = 6.5;
    rows.forEach((row, ri) => {
      if (yRef.v + RH + 2 > PH - 18) { yRef.v = newPage(); }
      const bg = ri === 0 ? C.accent : (ri % 2 === 1 ? C.rowAlt : C.pageBg);
      pdf.setFillColor(...bg); pdf.rect(M, yRef.v - 4.8, CW, RH, 'F');
      if (ri === 0) {
        pdf.setFont('helvetica', 'bold'); pdf.setFontSize(8); pdf.setTextColor(255, 255, 255);
      } else {
        pdf.setFont('helvetica', 'normal'); pdf.setFontSize(8); pdf.setTextColor(...C.text);
      }
      let cx = M + 2;
      row.forEach((cell, ci) => {
        if (ri > 0 && ci === row.length - 1) {
          const sc = cell === 'EXCEEDED' ? C.red : cell === 'ELEVATED' ? C.amber : cell === 'NORMAL' ? C.green : C.text;
          pdf.setFont('helvetica', ['EXCEEDED', 'ELEVATED', 'NORMAL'].includes(cell) ? 'bold' : 'normal');
          pdf.setTextColor(...sc);
        } else if (ri > 0 && ci === 0) {
          pdf.setFont('helvetica', 'bold'); pdf.setTextColor(...C.text);
        } else if (ri > 0) {
          pdf.setFont('helvetica', 'normal'); pdf.setTextColor(...C.text);
        }
        pdf.text(String(cell ?? '—'), cx + 1, yRef.v);
        cx += colW[ci];
      });
      yRef.v += RH;
    });
    yRef.v += 3;
  }

  function statBadges(items, yRef) {
    if (yRef.v + 18 > PH - 18) { yRef.v = newPage(); }
    const bw = CW / items.length - 2, bh = 16;
    items.forEach((it, i) => {
      const bx = M + i * (bw + 2);
      pdf.setFillColor(...C.accentLt); pdf.setDrawColor(...C.border);
      pdf.roundedRect(bx, yRef.v, bw, bh, 2, 2, 'FD');
      pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7); pdf.setTextColor(...C.muted);
      pdf.text(it.label, bx + bw / 2, yRef.v + 5.5, { align: 'center' });
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(13); pdf.setTextColor(...(it.color || C.accent));
      pdf.text(String(it.value ?? '—'), bx + bw / 2, yRef.v + 13, { align: 'center' });
    });
    yRef.v += bh + 5;
  }

  function signatureBlock(yRef) {
    if (yRef.v + 35 > PH - 18) { yRef.v = newPage(); }
    yRef.v += 8;
    pdf.setDrawColor(...C.border); pdf.setLineWidth(0.4);
    pdf.line(M, yRef.v, M + 72, yRef.v);
    pdf.line(M + 108, yRef.v, M + 180, yRef.v);
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7.5); pdf.setTextColor(...C.muted);
    pdf.text('Flight Surgeon / Mission Medical Officer', M, yRef.v + 4);
    pdf.text('Mission Commander', M + 108, yRef.v + 4);
    yRef.v += 12;
    pdf.text('Date: ___________________________', M, yRef.v);
    pdf.text('Date: ___________________________', M + 108, yRef.v);
  }

  return { C, riskColor, footer, newPage, section, kvRow, divider, para, tbl, statBadges, signatureBlock };
}

// ── Cover page helper ─────────────────────────────────────────────────────
function _drawCover(pdf, PW, PH, M, C, title, subtitle, meta, badgeText, badgeColor, ts) {
  pdf.setFillColor(...C.hdrDark); pdf.rect(0, 0, PW, 52, 'F');
  pdf.setFillColor(...C.hdrCyan); pdf.rect(0, 50, PW, 2, 'F');
  pdf.setFont('helvetica', 'bold'); pdf.setFontSize(24); pdf.setTextColor(...C.hdrCyan);
  pdf.text('ASTRONAUT DIGITAL TWIN', PW / 2, 20, { align: 'center' });
  pdf.setFontSize(12); pdf.setTextColor(175, 210, 240);
  pdf.text(title, PW / 2, 30, { align: 'center' });
  pdf.setFontSize(8); pdf.setTextColor(100, 145, 175);
  pdf.text(subtitle, PW / 2, 39, { align: 'center' });
  pdf.setFontSize(7.5); pdf.setTextColor(70, 110, 145);
  pdf.text(ts.toLocaleString(), PW / 2, 47, { align: 'center' });

  let y = 64;
  const CW = PW - M * 2;
  pdf.setFillColor(...C.accentLt); pdf.setDrawColor(...C.accent); pdf.setLineWidth(0.4);
  pdf.roundedRect(M, y, CW, 26, 2, 2, 'FD');
  pdf.setFont('helvetica', 'bold'); pdf.setFontSize(8.5); pdf.setTextColor(...C.accent);
  pdf.text('REPORT METADATA', M + 5, y + 7);
  let mx = y + 14;
  meta.forEach(([k, v], i) => {
    const col = i % 2 === 0 ? M + 5 : M + CW / 2;
    if (i % 2 === 0 && i > 0) mx += 5.5;
    pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7.5); pdf.setTextColor(...C.muted);
    pdf.text(k + ':', col, mx);
    pdf.setFont('helvetica', 'bold'); pdf.setTextColor(25, 30, 38);
    pdf.text(String(v), col + 32, mx);
    if (i % 2 === 1) mx += 5.5;
  });
  y += 34;

  if (badgeText) {
    pdf.setFillColor(...badgeColor);
    pdf.roundedRect(M, y, CW, 18, 3, 3, 'F');
    pdf.setFont('helvetica', 'bold'); pdf.setFontSize(16); pdf.setTextColor(255, 255, 255);
    pdf.text(badgeText, PW / 2, y + 11, { align: 'center' });
    y += 26;
  }

  pdf.setFont('helvetica', 'normal'); pdf.setFontSize(7.5); pdf.setTextColor(...C.muted);
  pdf.text('This report is auto-generated by the Astronaut Digital Twin platform.', PW / 2, PH - 20, { align: 'center' });
  pdf.text('Verify all values against raw simulation data before mission-critical decisions.', PW / 2, PH - 15, { align: 'center' });
  return y;
}

// ── Render a full time-series chart from raw arrays onto an offscreen canvas ──
function _renderFullChart(timeArr, dataArr, opts) {
  const W = opts.width || 600, H = opts.height || 200;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, W, H);
  const ml = 42, mr = 16, mt = 12, mb = 30;
  const pw = W - ml - mr, ph = H - mt - mb;
  const yMin = opts.yMin ?? 0, yMax = opts.yMax ?? 10;
  const n = timeArr.length;
  if (n < 2) return null;

  const gridLines = 5;
  ctx.strokeStyle = '#e0e8f0'; ctx.lineWidth = 0.8;
  for (let i = 0; i <= gridLines; i++) {
    const gy = mt + ph * (1 - i / gridLines);
    ctx.beginPath(); ctx.moveTo(ml, gy); ctx.lineTo(ml + pw, gy); ctx.stroke();
    const val = (yMin + (yMax - yMin) * i / gridLines).toFixed(yMax <= 1 ? 1 : 0);
    ctx.fillStyle = '#8090a0'; ctx.font = '10px Arial'; ctx.textAlign = 'right';
    ctx.fillText(val, ml - 4, gy + 3.5);
  }

  ctx.strokeStyle = '#b0bcc8'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ml, mt); ctx.lineTo(ml, mt + ph); ctx.lineTo(ml + pw, mt + ph); ctx.stroke();

  const tHours = timeArr.map(t => t / 60);
  const tMax = tHours[tHours.length - 1], tMin = tHours[0];
  const nXticks = Math.min(8, n);
  ctx.fillStyle = '#8090a0'; ctx.font = '10px Arial'; ctx.textAlign = 'center';
  for (let i = 0; i <= nXticks; i++) {
    const tf = tMin + (tMax - tMin) * i / nXticks;
    const px2 = ml + pw * i / nXticks;
    ctx.fillText(Math.round(tf) + 'h', px2, mt + ph + 16);
    ctx.strokeStyle = '#e0e8f0'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(px2, mt); ctx.lineTo(px2, mt + ph); ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(ml, mt + ph);
  for (let i = 0; i < n; i++) {
    const px2 = ml + pw * i / (n - 1);
    const py = mt + ph * (1 - Math.max(0, Math.min(1, (dataArr[i] - yMin) / (yMax - yMin))));
    ctx.lineTo(px2, py);
  }
  ctx.lineTo(ml + pw, mt + ph); ctx.closePath();
  ctx.fillStyle = (opts.color || '#0066cc') + '22'; ctx.fill();

  ctx.beginPath(); ctx.strokeStyle = opts.color || '#0066cc'; ctx.lineWidth = 1.5;
  for (let i = 0; i < n; i++) {
    const px2 = ml + pw * i / (n - 1);
    const py = mt + ph * (1 - Math.max(0, Math.min(1, (dataArr[i] - yMin) / (yMax - yMin))));
    i === 0 ? ctx.moveTo(px2, py) : ctx.lineTo(px2, py);
  }
  ctx.stroke();
  return c.toDataURL('image/png');
}

// ── BIOGEARS SIM REPORT ───────────────────────────────────────────────────
async function exportSimReport() {
  markUserInteraction();
  if (!currentAnalytics?.risk_report) {
    alert('No simulation data loaded. Run or load a BioGears simulation first.');
    return;
  }
  async function ensureScript(src, globalKey) {
    if (window[globalKey]) return;
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = () => rej(new Error('Failed to load ' + src));
      document.head.appendChild(s);
    });
  }
  const btn = document.getElementById('btn-export-report');
  if (btn) { btn.disabled = true; btn.textContent = '⟳ GENERATING PDF...'; }

  try {
    await ensureScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js', 'jspdf');
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const PW = 210, PH = 297, M = 14, CW = PW - M * 2;
    const ts = new Date();
    const pn = { n: 1 };
    const h = _buildPdfHelpers(pdf, PW, PH, M, CW, pn);
    const C = h.C;
    const y = { v: 0 };

    const rr   = currentAnalytics.risk_report;
    const th   = rr.threshold_metrics || {};
    const fat  = th.fatigue       || {};
    const slp  = th.sleep_quality || {};
    const cd   = rr.coupling_diagnostics;
    const lvl  = rr.overall_risk_level || 'UNKNOWN';
    const lvlC = h.riskColor(lvl);
    const missionDays = (Number(simSummary?.duration_hours || 0) / 1440).toFixed(1);

    // ── Cover ─────────────────────────────────────────────────────
    _drawCover(pdf, PW, PH, M, C,
      'BIOGEARS SIMULATION REPORT',
      'Coupled Sleep–Fatigue & Space Motion Sickness Model  ·  v2.0',
      [['Run ID', currentRunId || '—'], ['Generated', ts.toLocaleString()],
       ['Mission Duration', missionDays > 0 ? `${missionDays} days` : 'N/A'], ['Risk Level', lvl]],
      `OVERALL MISSION RISK:  ${lvl}`, lvlC, ts
    );
    y.v = 120;
    h.statBadges([
      { label: 'PEAK FATIGUE', value: fat.peak?.toFixed(1) ?? 'N/A', color: fat.peak > 7 ? C.red : fat.peak > 5 ? C.amber : C.green },
      { label: 'MEAN FATIGUE', value: fat.mean?.toFixed(1) ?? 'N/A', color: fat.mean > 5 ? C.amber : C.green },
      { label: 'MEAN SLEEP Q', value: slp.mean?.toFixed(2) ?? 'N/A', color: slp.mean < 0.5 ? C.red : slp.mean < 0.7 ? C.amber : C.green },
      { label: 'RISK WINDOWS', value: rr.n_risk_windows ?? 'N/A', color: (rr.n_risk_windows || 0) > 3 ? C.amber : C.green },
    ], y);
    h.footer('BIOGEARS SIMULATION REPORT');

    // ── Page 2: Summary ───────────────────────────────────────────
    pdf.addPage(); pn.n++;
    pdf.setFillColor(...C.pageBg); pdf.rect(0, 0, PW, PH, 'F');
    pdf.setFillColor(...C.accent); pdf.rect(0, 0, PW, 1.5, 'F');
    y.v = 14;

    h.section('1.  EXECUTIVE SUMMARY', y);
    h.kvRow('Overall Risk Level',       lvl, lvlC, lvl, lvlC, y);
    h.kvRow('Mission Duration',         missionDays > 0 ? `${missionDays} days` : 'N/A', null, null, null, y);
    h.kvRow('Peak Fatigue  (0–10)',      fat.peak?.toFixed(2) ?? 'N/A', fat.peak > 7 ? C.red : fat.peak > 5 ? C.amber : C.green, fat.peak > 7 ? 'EXCEEDED' : fat.peak > 5 ? 'ELEVATED' : 'NORMAL', fat.peak > 7 ? C.red : fat.peak > 5 ? C.amber : C.green, y);
    h.kvRow('Mean Fatigue',             fat.mean?.toFixed(2) ?? 'N/A', null, null, null, y);
    h.kvRow('Time Above Fatigue Thr.',  fat.time_above_threshold != null ? fat.time_above_threshold.toFixed(1) + ' hrs' : 'N/A', null, null, null, y);
    h.kvRow('Mean Sleep Quality (0–1)', slp.mean?.toFixed(3) ?? 'N/A', slp.mean < 0.5 ? C.red : slp.mean < 0.7 ? C.amber : C.green, null, null, y);
    h.kvRow('P(Critical Sleep)',        slp.prob_critical != null ? `${(slp.prob_critical * 100).toFixed(1)}%` : 'N/A', null, null, null, y);
    h.kvRow('P(Poor Sleep)',            slp.prob_poor != null ? `${(slp.prob_poor * 100).toFixed(1)}%` : 'N/A', null, null, null, y);
    h.kvRow('At-Risk Windows Detected', rr.n_risk_windows ?? 'N/A', (rr.n_risk_windows || 0) > 3 ? C.amber : C.text, null, null, y);
    h.divider(y);

    // ── Full time-series charts ───────────────────────────────────
    if (simResults?.state) {
      const st = simResults.state;
      h.section('2.  PHYSIOLOGICAL TIME-SERIES  (FULL MISSION)', y);
      const chartH = 44;
      const halfW = (CW - 4) / 2;

      function addChartPair(img1, label1, img2, label2) {
        if (!img1 && !img2) return;
        if (y.v + chartH + 8 > PH - 18) { y.v = h.newPage('BIOGEARS SIMULATION REPORT'); }
        pdf.setFont('helvetica', 'bold'); pdf.setFontSize(7.5); pdf.setTextColor(...C.muted);
        if (img1) { pdf.text(label1, M, y.v); pdf.addImage(img1, 'PNG', M, y.v + 3, img2 ? halfW : CW, chartH); }
        if (img2) { pdf.text(label2, M + halfW + 4, y.v); pdf.addImage(img2, 'PNG', M + halfW + 4, y.v + 3, halfW, chartH); }
        y.v += chartH + 8;
      }

      const fatImg = _renderFullChart(st.time, st.fatigue,       { color: '#ffaa00', yMin: 0, yMax: 10,  width: 700, height: 220 });
      const slpImg = _renderFullChart(st.time, st.sleep_quality, { color: '#00cc66', yMin: 0, yMax: 1,   width: 700, height: 220 });
      const hrImg  = _renderFullChart(st.time, st.hr,            { color: '#ff8833', yMin: 40, yMax: 160, width: 700, height: 220 });
      const rskData = st.fatigue.map(f => Math.min(1, f / 10));
      const rskImg  = _renderFullChart(st.time, rskData,         { color: '#ee3333', yMin: 0, yMax: 1,   width: 700, height: 220 });

      addChartPair(fatImg, 'FATIGUE INDEX (0–10)', slpImg, 'SLEEP QUALITY (0–1)');
      addChartPair(hrImg,  'HEART RATE (bpm)',     rskImg, 'RISK INDEX (0–1)');
      h.divider(y);
    }

    // ── Fatigue Analysis ──────────────────────────────────────────
    const s3 = simResults?.state ? '3' : '2';
    h.section(`${s3}.  FATIGUE ANALYSIS`, y);
    h.tbl([
      ['Metric', 'Value', 'Threshold', 'Status'],
      ['Peak Fatigue',         fat.peak?.toFixed(2) ?? '—', '7.0  (HIGH)', fat.peak > 7 ? 'EXCEEDED' : fat.peak > 5 ? 'ELEVATED' : 'NORMAL'],
      ['Mean Fatigue',         fat.mean?.toFixed(2) ?? '—', '5.0  (MOD)',  fat.mean > 5 ? 'EXCEEDED' : 'NORMAL'],
      ['Time Above Threshold', fat.time_above_threshold != null ? fat.time_above_threshold.toFixed(1) + ' hrs' : '—', '—', '—'],
      ['Fraction Above Thr.',  fat.frac_above_threshold != null ? (fat.frac_above_threshold * 100).toFixed(1) + '%' : '—', '—', '—'],
    ], [60, 38, 46, CW - 144], y);

    // ── Sleep Quality ─────────────────────────────────────────────
    h.section(`${+s3 + 1}.  SLEEP QUALITY ANALYSIS`, y);
    h.tbl([
      ['Metric', 'Value', 'Status'],
      ['Mean Sleep Quality',     slp.mean?.toFixed(3) ?? '—', slp.mean < 0.5 ? 'EXCEEDED' : slp.mean < 0.7 ? 'ELEVATED' : 'NORMAL'],
      ['Min Sleep Quality',      slp.min?.toFixed(3) ?? '—',  '—'],
      ['P(Critical Sleep < 0.4)', slp.prob_critical != null ? (slp.prob_critical * 100).toFixed(1) + '%' : '—', '—'],
      ['P(Poor Sleep < 0.6)',    slp.prob_poor != null ? (slp.prob_poor * 100).toFixed(1) + '%' : '—', '—'],
      ['Time in Critical Sleep', slp.time_critical != null ? slp.time_critical.toFixed(1) + ' hrs' : '—', '—'],
    ], [80, 50, CW - 130], y);
    h.divider(y);

    // ── Risk Windows ──────────────────────────────────────────────
    h.section(`${+s3 + 2}.  RISK WINDOWS & EVENTS`, y);
    const riskWindows = rr.risk_windows || [];
    if (riskWindows.length === 0) {
      h.para('No specific risk windows identified in this simulation run.', 2, y);
    } else {
      riskWindows.slice(0, 15).forEach((w, i) => {
        if (y.v + 7 > PH - 18) { y.v = h.newPage('BIOGEARS SIMULATION REPORT'); }
        const wc = h.riskColor(w.risk_level);
        pdf.setFillColor(...C.rowAlt); pdf.rect(M, y.v - 4.5, CW, 6.5, 'F');
        pdf.setFillColor(...wc); pdf.rect(M, y.v - 4.5, 2.5, 6.5, 'F');
        pdf.setFont('helvetica', 'bold'); pdf.setFontSize(7.5); pdf.setTextColor(...C.accent);
        pdf.text(`W${i + 1}`, M + 4, y.v);
        pdf.setFont('helvetica', 'normal'); pdf.setTextColor(...C.text);
        pdf.text(`T+${(w.start_hour || 0).toFixed(0)}h – T+${(w.end_hour || 0).toFixed(0)}h`, M + 13, y.v);
        pdf.setFont('helvetica', 'bold'); pdf.setTextColor(...wc);
        pdf.text(w.risk_level || '', M + 45, y.v);
        pdf.setFont('helvetica', 'normal'); pdf.setTextColor(...C.text);
        const desc = pdf.splitTextToSize(w.description || '', CW - 75);
        pdf.text(desc[0] || '', M + 75, y.v);
        y.v += 6.5;
      });
      if (riskWindows.length > 15) h.para(`… and ${riskWindows.length - 15} additional windows in raw data.`, 2, y);
    }
    h.divider(y);

    // ── Coupling Diagnostics ──────────────────────────────────────
    if (cd) {
      h.section(`${+s3 + 3}.  VESTIBULAR–FATIGUE COUPLING`, y);
      h.kvRow('Mean Excess P(MS)',            cd.mean_excess_p_ms?.toFixed(4) ?? '—', null, null, null, y);
      h.kvRow('Joint Risk Excess Fraction',   cd.joint_risk_excess != null ? (cd.joint_risk_excess * 100).toFixed(1) + '%' : '—', null, null, null, y);
      h.kvRow('Mean Suppression Factor kₛ',   cd.mean_k_suppress?.toFixed(3) ?? '—', null, null, null, y);
      h.kvRow('Time in High-Coupling Regime', cd.time_high_coupling_frac != null ? (cd.time_high_coupling_frac * 100).toFixed(1) + '%' : '—', null, null, null, y);
      h.divider(y);
    }

    // ── Conclusions ───────────────────────────────────────────────
    const cn = +s3 + 2 + (riskWindows.length >= 0 ? 1 : 0) + (cd ? 1 : 0) + 1;
    h.section(`${cn}.  CONCLUSIONS & RECOMMENDATIONS`, y);
    const concEl = document.getElementById('mission-risk-conclusions');
    if (concEl?.innerText.trim()) {
      concEl.innerText.trim().split('\n').filter(Boolean).forEach(l => h.para(l, 2, y));
    } else {
      const lines = [];
      if (fat.peak > 7)       lines.push('CRITICAL: Peak fatigue exceeded 7.0 — immediate crew rest protocol required.');
      else if (fat.peak > 5)  lines.push('WARNING: Peak fatigue elevated above 5.0 — monitor and pre-position rest windows.');
      else if (fat.peak)      lines.push('Fatigue levels remained within acceptable limits throughout the mission.');
      if (slp.mean < 0.5)     lines.push('CRITICAL: Mean sleep quality below 0.5 — sleep hygiene intervention required.');
      else if (slp.mean < 0.7) lines.push('WARNING: Mean sleep quality sub-optimal — review sleep schedule and light exposure.');
      else if (slp.mean)      lines.push('Sleep quality maintained at acceptable levels.');
      if ((rr.n_risk_windows || 0) > 3) lines.push(`${rr.n_risk_windows} at-risk windows detected — cross-reference with EVA schedule.`);
      lines.push('All findings should be reviewed by the flight surgeon before mission go/no-go decision.');
      lines.forEach(l => h.para('• ' + l, 2, y));
    }
    h.signatureBlock(y);
    h.footer('BIOGEARS SIMULATION REPORT');

    const filename = `ADT_SimReport_${(currentRunId || 'sim').slice(0, 12)}_${ts.toISOString().slice(0, 10)}.pdf`;
    pdf.save(filename);
  } catch (e) {
    console.error('Sim report export failed:', e);
    alert('Export failed: ' + e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'EXPORT REPORT'; }
  }
}

// ── MONTE CARLO REPORT ────────────────────────────────────────────────────
async function exportMCReport() {
  markUserInteraction();
  const mcStatEl = document.getElementById('mc-stats');
  if (!mcStatEl?.innerText.trim()) {
    alert('No Monte Carlo data available. Run Monte Carlo first.');
    return;
  }
  async function ensureScript(src, globalKey) {
    if (window[globalKey]) return;
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = res; s.onerror = () => rej(new Error('Failed to load ' + src));
      document.head.appendChild(s);
    });
  }
  const btnMC = document.getElementById('btn-export-report-mc');
  if (btnMC) { btnMC.disabled = true; btnMC.textContent = '⟳ GENERATING PDF...'; }

  try {
    await ensureScript('https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js', 'jspdf');
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const PW = 210, PH = 297, M = 14, CW = PW - M * 2;
    const ts = new Date();
    const pn = { n: 1 };
    const h = _buildPdfHelpers(pdf, PW, PH, M, CW, pn);
    const C = h.C;
    const y = { v: 0 };

    const statCards = Array.from(mcStatEl.querySelectorAll('.mcstat'));
    const badges = statCards.map(c => ({
      label: (c.querySelector('.mcslabel')?.textContent || '').trim(),
      value: (c.querySelector('.mcsval')?.textContent || '').trim(),
      color: c.querySelector('.mcsval.R') ? C.red : c.querySelector('.mcsval.W') ? C.amber : C.accent,
    }));
    const nRuns    = badges.find(b => b.label === 'n RUNS')?.value ?? '—';
    const pFatRisk = badges.find(b => b.label.includes('FAT RISK'))?.value ?? '—';
    const p95Peak  = badges.find(b => b.label.includes('P95'))?.value ?? '—';
    const medRec   = badges.find(b => b.label.includes('RECOVERY'))?.value ?? '—';

    // ── Cover ─────────────────────────────────────────────────────
    _drawCover(pdf, PW, PH, M, C,
      'MONTE CARLO PROBABILISTIC REPORT',
      'Coupled Sleep–Fatigue & Space Motion Sickness Model  ·  v2.0',
      [['Simulation Runs', nRuns], ['Generated', ts.toLocaleString()],
       ['P(Fatigue Risk)', pFatRisk], ['P95 Peak Fatigue', p95Peak]],
      null, null, ts
    );
    y.v = 108;
    if (badges.length >= 4) h.statBadges(badges.slice(0, 4), y);
    h.footer('MONTE CARLO PROBABILISTIC REPORT');

    // ── Page 2: Charts + Results ───────────────────────────────────
    pdf.addPage(); pn.n++;
    pdf.setFillColor(...C.pageBg); pdf.rect(0, 0, PW, PH, 'F');
    pdf.setFillColor(...C.accent); pdf.rect(0, 0, PW, 1.5, 'F');
    y.v = 14;

    h.section('1.  SIMULATION PARAMETERS', y);
    h.kvRow('Number of Runs',       nRuns, null, null, null, y);
    h.kvRow('Median Recovery Time', medRec, null, null, null, y);
    h.kvRow('P95 Peak Fatigue',     p95Peak, parseFloat(p95Peak) >= 7 ? C.red : C.amber, null, null, y);
    h.kvRow('P(Fatigue Risk) Mean', pFatRisk, parseFloat(pFatRisk) >= 30 ? C.amber : C.green, null, null, y);
    h.divider(y);

    // ── 2. Probabilistic Envelopes ────────────────────────────────
    const mcFImg = (() => { const c = document.getElementById('ch-mc-t'); return c && c.width > 0 ? c.toDataURL('image/png') : null; })();
    const mcSImg = (() => { const c = document.getElementById('ch-mc-h'); return c && c.width > 0 ? c.toDataURL('image/png') : null; })();

    if (mcFImg || mcSImg) {
      h.section('2.  PROBABILISTIC ENVELOPES  (ALL RUNS)', y);
      const halfW = (CW - 4) / 2, chartH = 52;
      if (y.v + chartH + 10 > PH - 18) { y.v = h.newPage('MONTE CARLO PROBABILISTIC REPORT'); }
      pdf.setFont('helvetica', 'bold'); pdf.setFontSize(7.5); pdf.setTextColor(...C.muted);
      if (mcFImg) { pdf.text('FATIGUE ENVELOPE (mean ± range)', M, y.v); pdf.addImage(mcFImg, 'PNG', M, y.v + 3, mcSImg ? halfW : CW, chartH); }
      if (mcSImg) { pdf.text('SLEEP QUALITY ENVELOPE (mean ± 1σ)', M + halfW + 4, y.v); pdf.addImage(mcSImg, 'PNG', M + halfW + 4, y.v + 3, halfW, chartH); }
      y.v += chartH + 10;
      h.divider(y);
    }

    // ── 3. Risk Analysis Narrative ────────────────────────────────
    h.section('3.  PROBABILISTIC RISK FINDINGS', y);
    const mcConcEl = document.getElementById('mc-conclusions');
    if (mcConcEl?.innerText.trim()) {
      mcConcEl.innerText.trim().split('\n').filter(l => l.trim()).forEach(l => h.para(l.trim(), 2, y));
    } else {
      h.para('Monte Carlo conclusions not available. Run the simulation to generate findings.', 2, y);
    }
    h.divider(y);

    // ── 4. Interpretation ─────────────────────────────────────────
    h.section('4.  CLINICAL INTERPRETATION & RECOMMENDATIONS', y);
    const pFat = parseFloat(pFatRisk) || 0;
    const p95  = parseFloat(p95Peak)  || 0;
    const interp = [];
    if (p95 >= 9)       interp.push('CRITICAL: P95 peak fatigue ≥9.0/10 — a significant fraction of missions will reach incapacitating fatigue levels. Mandatory rest blocks and workload redistribution required before mission approval.');
    else if (p95 >= 7)  interp.push('WARNING: P95 peak fatigue ≥7.0/10 — high-fatigue outcomes are probable in the worst-case trajectory. Proactive fatigue countermeasures (napping, stimulants) should be pre-positioned.');
    else                interp.push('P95 peak fatigue is within manageable bounds. Standard fatigue monitoring protocols apply.');
    if (pFat >= 50)     interp.push('CRITICAL: Over 50% of simulated missions breach the fatigue risk threshold — the mission profile is high-risk by design. Mission replanning is strongly recommended.');
    else if (pFat >= 25) interp.push('WARNING: Approx. ' + pFatRisk + ' of missions exceed the fatigue threshold. Sleep scheduling optimisation and early intervention protocols should be activated.');
    else                interp.push('Fatigue risk probability is acceptable. Continue standard health monitoring.');
    interp.push('All probabilistic findings should be reviewed alongside deterministic BioGears simulation results by the flight surgeon before mission go/no-go decision.');
    interp.forEach(l => h.para('• ' + l, 2, y));
    h.signatureBlock(y);
    h.footer('MONTE CARLO PROBABILISTIC REPORT');

    const filename = `ADT_MCReport_${ts.toISOString().slice(0, 10)}.pdf`;
    pdf.save(filename);
  } catch (e) {
    console.error('MC report export failed:', e);
    alert('Export failed: ' + e.message);
  } finally {
    if (btnMC) { btnMC.disabled = false; btnMC.textContent = 'EXPORT REPORT'; }
  }
}