// static/js/tour.js
// ════════════════════════════════════════════════════════
// ONBOARDING TOUR
// ════════════════════════════════════════════════════════

const TOUR_STEPS = [
  { sel: '#panel-ctrl', title: 'MISSION CONTROLS', desc: 'Configure your mission parameters here — duration, SMS rate, fatigue sensitivity, and more.' },
  { sel: '#btn-run', title: 'LAUNCH SIMULATION', desc: 'Start the BioGears physiological simulation. Your astronaut digital twin will process in real-time.' },
  { sel: '#panel-twin', title: '3D DIGITAL TWIN', desc: 'Watch your astronaut\'s physiological state in real time during playback. Drag to orbit, scroll to zoom, click the astronaut to inspect vitals.' },
  { sel: '#tl-section', title: 'TIMELINE + CHARTS', desc: 'Scrub through the full mission timeline. Each day shows activity pills and fatigue indicators. Expand charts for full-screen analysis.' },
  { sel: '#mc-section', title: 'MONTE CARLO ANALYSIS', desc: 'Run probabilistic risk analysis across hundreds of trajectories to understand worst-case scenarios.' },
  { sel: '#btn-load', title: 'LOAD PREVIOUS RUNS', desc: 'Load and compare completed simulation runs. See how different parameters affect mission outcomes.' },
];
let tourIndex = 0;

function _tourPlaceFor(step) {
  const overlay = document.getElementById('tour-overlay');
  const cut = document.getElementById('tour-cutout');
  const tip = document.getElementById('tour-tip');
  const ttl = document.getElementById('tour-tip-title');
  const desc = document.getElementById('tour-tip-desc');
  const next = document.getElementById('tour-next-btn');
  if (!overlay || !cut || !tip || !ttl || !desc || !next) return;
  
  const target = document.querySelector(step.sel);
  if (!target || target.offsetParent === null) {
    // Try to make hidden sections visible temporarily
    if (step.sel === '#tl-section') document.getElementById('tl-section').style.display = 'block';
    if (step.sel === '#mc-section') document.getElementById('mc-section').style.display = 'block';
  }
  
  const t2 = document.querySelector(step.sel);
  if (!t2) return;
  
  t2.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const r = t2.getBoundingClientRect();
  
  cut.style.left = `${Math.max(0, r.left - 6)}px`;
  cut.style.top = `${Math.max(0, r.top - 6)}px`;
  cut.style.width = `${Math.max(30, r.width + 12)}px`;
  cut.style.height = `${Math.max(30, r.height + 12)}px`;
  
  ttl.textContent = step.title;
  desc.textContent = step.desc;
  next.textContent = tourIndex === TOUR_STEPS.length - 1 ? 'DONE' : 'NEXT';
  
  const tipTop = Math.min(window.innerHeight - tip.offsetHeight - 8, r.bottom + 10);
  const tipLeft = Math.min(window.innerWidth - tip.offsetWidth - 8, Math.max(8, r.left));
  tip.style.top = `${tipTop}px`;
  tip.style.left = `${tipLeft}px`;
}

function closeTour(setFlag = true) {
  const overlay = document.getElementById('tour-overlay');
  if (overlay) overlay.style.display = 'none';
  if (setFlag) localStorage.setItem(TOURED_KEY, '1');
  
  // Hide sections that were temporarily shown
  if (!simResults) {
    document.getElementById('tl-section').style.display = 'none';
    document.getElementById('mc-section').style.display = 'none';
  }
}

function tourNext() {
  tourIndex += 1;
  if (tourIndex >= TOUR_STEPS.length) {
    closeTour(true);
    return;
  }
  _tourPlaceFor(TOUR_STEPS[tourIndex]);
}

function startTour(force = false) {
  if (force) markUserInteraction();
  if (!force && localStorage.getItem(TOURED_KEY) === '1') return;
  
  const overlay = document.getElementById('tour-overlay');
  const next = document.getElementById('tour-next-btn');
  const skip = document.getElementById('tour-skip-btn');
  if (!overlay || !next || !skip) return;
  
  overlay.style.display = 'block';
  tourIndex = 0;
  _tourPlaceFor(TOUR_STEPS[tourIndex]);
  
  next.onclick = tourNext;
  skip.onclick = () => closeTour(true);
}

window.addEventListener('resize', () => {
  const overlay = document.getElementById('tour-overlay');
  if (overlay && overlay.style.display === 'block') _tourPlaceFor(TOUR_STEPS[tourIndex]);
});