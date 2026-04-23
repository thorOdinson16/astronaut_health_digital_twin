// static/js/3d-viewer.js
// ════════════════════════════════════════════════════════
// THREE.JS — ENDURANCE SPACESHIP + HABITAT INTERIOR
// ════════════════════════════════════════════════════════

let renderer3, scene3, cam3, astronaut, aimb;
let tarsGroup, tarsLight, tarsScreenMat;
let bodyMesh, bodyMat;
let riskRing, riskRingMat;
let enduranceRing;          // the rotating ring
let roomContainer;          // room + astronaut + TARS parented group
let astroLookTarget;        // camera look-at for astronaut view
let threeOK = false;
let glbLoaded = false;

const _tmpV1 = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();

// Audio context for sound effects
let audioCtx = null;
let muted = false;
let prevRiskLevel = 'NOMINAL';
let userInteractedForAudio = false;

let inspectDismissTimer = null;
let inspectRaycaster = null;
let inspectPointer = null;
let inspectDragMoved = false;
let inspectDownPoint = null;
let inspectDownTs = 0;

let habitatLight = null;
let ambientBaseColor = null;
let habitatBaseColor = null;
let highRiskLighting = false;
let tremorX = 0;
let tremorZ = 0;

const bgm = document.getElementById('bgm');

// Camera orbit state
const camState = {
  radius: 110,
  theta: 0.4,
  phi: 0.22,
  targetY: 0,
  isDragging: false,
  lastX: 0, lastY: 0,
  autoRotate: true,
  minRadius: 2.5,
  maxRadius: 1000,
  lastPinchDist: null,
};

let camMode = 'ship';

// Default astronaut camera position
const astroCamDefault = { 
  radius: 5.6, 
  theta: 0.08, 
  phi: 0.0873, 
  targetY: 1.8, 
  lookX: -0.3, 
  lookY: 0.0, 
  lookZ: -4.1 
};

let _t3last = 0, _simState = null;

// ── AUDIO FUNCTIONS ────────────────────────────────────────────

function ensureAudio() {
  if (audioCtx) return audioCtx;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return null;
  audioCtx = new Ctx();
  return audioCtx;
}

function markUserInteraction() {
  userInteractedForAudio = true;
  const ctx = ensureAudio();
  if (ctx && ctx.state === 'suspended') ctx.resume().catch(() => {});
}

function _playTone(freq, duration, gain = 0.05) {
  const ctx = ensureAudio();
  if (!ctx || muted || !userInteractedForAudio) return;
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  const osc = ctx.createOscillator();
  const g = ctx.createGain();
  osc.type = 'sine';
  osc.frequency.setValueAtTime(freq, ctx.currentTime);
  g.gain.setValueAtTime(0.0001, ctx.currentTime);
  g.gain.linearRampToValueAtTime(gain, ctx.currentTime + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
  osc.connect(g);
  g.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + duration + 0.02);
}

function playRiskTransition(prevLevel, nextLevel) {
  if (muted) return;
  if (prevLevel === 'MODERATE' && nextLevel === 'HIGH') {
    _playTone(220, 0.16, 0.045);
  } else if (prevLevel === 'HIGH' && nextLevel === 'CRITICAL') {
    _playTone(300, 0.12, 0.05);
    setTimeout(() => _playTone(460, 0.18, 0.055), 140);
  }
}

function toggleMute() {
  markUserInteraction();
  muted = !muted;
  const btn = document.getElementById('btn-mute');
  if (!btn) return;
  btn.textContent = muted ? '🔇 MUTED' : '🔊 SOUND';
  btn.style.borderColor = muted ? 'var(--amber)' : 'var(--border2)';
  btn.style.color = muted ? 'var(--amber)' : 'var(--text)';
  
  if (bgm) {
    if (muted) {
      bgm.pause();
    } else if (playState && playState.running && !playState.paused) {
      try { bgm.play(); } catch(e) { console.warn('BGM play blocked:', e); }
    }
  }
}

function fadeOutBGM(duration = 3000) {
  if (!bgm || bgm.paused) return;
  const startVol = bgm.volume;
  const startTime = performance.now();
  function tick() {
    const elapsed = performance.now() - startTime;
    const progress = Math.min(elapsed / duration, 1);
    bgm.volume = startVol * (1 - progress);
    if (progress < 1) requestAnimationFrame(tick);
    else { bgm.pause(); bgm.currentTime = 0; bgm.volume = 1; }
  }
  requestAnimationFrame(tick);
}

// ── INSPECT TOOLTIP ────────────────────────────────────────────

function hideInspectTip() {
  clearTimeout(inspectDismissTimer);
  const tip = document.getElementById('inspect-tip');
  if (tip) tip.style.display = 'none';
}

function showInspectTip(clientX, clientY, title, valueHtml) {
  const panel = document.getElementById('panel-twin');
  const tip = document.getElementById('inspect-tip');
  if (!panel || !tip) return;
  const rect = panel.getBoundingClientRect();
  tip.querySelector('.ititle').textContent = title;
  tip.querySelector('.ival').innerHTML = valueHtml;
  tip.style.display = 'block';
  const x = Math.max(6, Math.min(rect.width - tip.offsetWidth - 6, clientX - rect.left + 10));
  const y = Math.max(6, Math.min(rect.height - tip.offsetHeight - 6, clientY - rect.top + 10));
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
  clearTimeout(inspectDismissTimer);
  inspectDismissTimer = setTimeout(hideInspectTip, 3000);
}

function _resolveAstronautMeshes() {
  const meshes = [];
  if (!astronaut) return meshes;
  astronaut.traverse(obj => { if (obj.isMesh) meshes.push(obj); });
  return meshes;
}

function inspectAstronautAt(clientX, clientY) {
  if (!renderer3 || !cam3 || !astronaut || !_simState) return;
  const canvas = renderer3.domElement;
  const rect = canvas.getBoundingClientRect();
  inspectPointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
  inspectPointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  inspectRaycaster.setFromCamera(inspectPointer, cam3);
  const hits = inspectRaycaster.intersectObjects(_resolveAstronautMeshes(), true);
  if (!hits.length) { hideInspectTip(); return; }

  const hit = hits[0];
  const box = new THREE.Box3().setFromObject(astronaut);
  const ySpan = Math.max(0.001, box.max.y - box.min.y);
  const yNorm = (hit.point.y - box.min.y) / ySpan;
  const s = _simState;
  if (yNorm >= 0.72) {
    showInspectTip(clientX, clientY, 'HEAD · STRESS', `STRESS: ${(s.stress || 0).toFixed(2)}`);
  } else if (yNorm >= 0.42) {
    showInspectTip(clientX, clientY, 'UPPER TORSO', `HEART RATE: ${(s.hr || 0).toFixed(0)} bpm<br>SpO₂: ${(s.spo2 || 0).toFixed(1)}%`);
  } else {
    showInspectTip(clientX, clientY, 'LOWER BODY', `FATIGUE: ${(s.fatigueIndex || 0).toFixed(2)}<br>MOTION SEVERITY: ${(s.motionSeverity || 0).toFixed(2)}`);
  }
}

function initInspectHandlers() {
  const cv = document.getElementById('three-canvas');
  if (!cv || !window.THREE) return;
  inspectRaycaster = new THREE.Raycaster();
  inspectPointer = new THREE.Vector2();

  cv.addEventListener('mousedown', (e) => {
    inspectDragMoved = false;
    inspectDownPoint = { x: e.clientX, y: e.clientY };
    inspectDownTs = performance.now();
  });
  
  window.addEventListener('mousemove', (e) => {
    if (!inspectDownPoint) return;
    if (Math.hypot(e.clientX - inspectDownPoint.x, e.clientY - inspectDownPoint.y) > 8) 
      inspectDragMoved = true;
  });
  
  window.addEventListener('mouseup', (e) => {
    if (!inspectDownPoint) return;
    const clickLike = !inspectDragMoved && (performance.now() - inspectDownTs < 380);
    const p = inspectDownPoint;
    inspectDownPoint = null;
    if (!clickLike) return;
    hideInspectTip();
    inspectAstronautAt(e.clientX || p.x, e.clientY || p.y);
  });

  document.addEventListener('mousedown', (e) => {
    if (!e.target.closest || !e.target.closest('#inspect-tip')) hideInspectTip();
  });
}

// ── TARS ROBOT ────────────────────────────────────────────

function buildTARS() {
  const group = new THREE.Group();
  const bodyMat = new THREE.MeshPhongMaterial({ color: 0x2a2d30, shininess: 180, specular: 0x999999 });
  const darkMat = new THREE.MeshPhongMaterial({ color: 0x111314, shininess: 40, specular: 0x333333 });
  const chromeMat = new THREE.MeshPhongMaterial({ color: 0x777c80, shininess: 255, specular: 0xffffff });

  const panelW = 0.28, panelH = 1.90, panelD = 0.22, gap = 0.038;
  const totalW = 4 * panelW + 3 * gap;
  const startX = -totalW / 2 + panelW / 2;

  for (let i = 0; i < 4; i++) {
    const px = startX + i * (panelW + gap);
    const panelGroup = new THREE.Group();
    panelGroup.position.set(px, 0, 0);

    const panel = new THREE.Mesh(new THREE.BoxGeometry(panelW, panelH, panelD), bodyMat.clone());
    panel.position.set(0, panelH / 2, 0);
    panelGroup.add(panel);

    const face = new THREE.Mesh(new THREE.BoxGeometry(panelW - 0.04, panelH - 0.04, 0.01), darkMat);
    face.position.set(0, panelH / 2, panelD / 2 + 0.005);
    panelGroup.add(face);

    const topCap = new THREE.Mesh(new THREE.BoxGeometry(panelW + 0.01, 0.04, panelD + 0.01), chromeMat);
    topCap.position.set(0, panelH + 0.02, 0);
    panelGroup.add(topCap);

    const botCap = new THREE.Mesh(new THREE.BoxGeometry(panelW + 0.01, 0.04, panelD + 0.01), chromeMat);
    botCap.position.set(0, -0.02, 0);
    panelGroup.add(botCap);

    const railGeo = new THREE.BoxGeometry(0.015, panelH, panelD + 0.02);
    const railL = new THREE.Mesh(railGeo, chromeMat);
    const railR = new THREE.Mesh(railGeo, chromeMat);
    railL.position.set(-panelW / 2 - 0.007, panelH / 2, 0);
    railR.position.set(panelW / 2 + 0.007, panelH / 2, 0);
    panelGroup.add(railL, railR);

    for (let s = 0; s < 5; s++) {
      const score = new THREE.Mesh(new THREE.BoxGeometry(panelW - 0.06, 0.008, 0.012), chromeMat);
      score.position.set(0, 0.3 + s * 0.32, panelD / 2 + 0.006);
      panelGroup.add(score);
    }

    if (i === 1) panelGroup.rotation.x = Math.PI / 18;
    if (i === 2) panelGroup.rotation.x = -Math.PI / 18;
    group.add(panelGroup);
  }

  const eyeY = panelH * 0.62;
  const eyeW = totalW + 0.02;

  const eyeSlot = new THREE.Mesh(new THREE.BoxGeometry(eyeW, 0.09, 0.04), darkMat);
  eyeSlot.position.set(0, eyeY, panelD / 2 - 0.01);
  group.add(eyeSlot);

  tarsScreenMat = new THREE.MeshBasicMaterial({ color: 0x00c8ff });
  const eye = new THREE.Mesh(new THREE.BoxGeometry(eyeW - 0.04, 0.045, 0.02), tarsScreenMat);
  eye.position.set(0, eyeY, panelD / 2 + 0.01);
  group.add(eye);

  const glowMat = new THREE.MeshBasicMaterial({ color: 0x004466, transparent: true, opacity: 0.4 });
  const eyeGlow = new THREE.Mesh(new THREE.BoxGeometry(eyeW + 0.1, 0.18, 0.005), glowMat);
  eyeGlow.position.set(0, eyeY, panelD / 2 + 0.02);
  group.add(eyeGlow);

  const scanMat = new THREE.MeshBasicMaterial({ color: 0x00ffee, transparent: true, opacity: 0.7 });
  const scan = new THREE.Mesh(new THREE.BoxGeometry(eyeW - 0.06, 0.012, 0.015), scanMat);
  scan.position.set(0, eyeY - 0.3, panelD / 2 + 0.015);
  scan.name = 'tars_scan';
  group.add(scan);

  const chromeMat2 = new THREE.MeshPhongMaterial({ color: 0x777c80, shininess: 255, specular: 0xffffff });
  const basePlate = new THREE.Mesh(new THREE.BoxGeometry(totalW + 0.12, 0.06, panelD + 0.08), chromeMat2);
  basePlate.position.set(0, -0.03, 0);
  group.add(basePlate);

  const footMat = new THREE.MeshPhongMaterial({ color: 0x1a1d1f, shininess: 60, specular: 0x555555 });
  const footL = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.18, 0.30), footMat);
  const footR = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.18, 0.30), footMat);
  footL.position.set(-totalW / 2 + 0.11, -0.09, 0.04);
  footR.position.set(totalW / 2 - 0.11, -0.09, 0.04);
  group.add(footL, footR);

  tarsLight = new THREE.PointLight(0x00c8ff, 1.2, 3.5);
  tarsLight.position.set(0, eyeY, panelD / 2 + 0.8);
  group.add(tarsLight);

  return group;
}

// ── SPACESHIP INTERIOR ROOM ───────────────────────────────

function buildSpaceshipInterior() {
  const group = new THREE.Group();

  const steelMat   = new THREE.MeshPhongMaterial({ color: 0x1c2128, shininess: 60, specular: 0x334455 });
  const darkSteel  = new THREE.MeshPhongMaterial({ color: 0x0e1218, shininess: 30, specular: 0x223344 });
  const panelMat   = new THREE.MeshPhongMaterial({ color: 0x161c22, shininess: 20, specular: 0x1a2233 });
  const chromeMat  = new THREE.MeshPhongMaterial({ color: 0x445566, shininess: 200, specular: 0xaabbcc });
  const glowMat    = new THREE.MeshBasicMaterial({ color: 0x001833 });
  const floorMat   = new THREE.MeshPhongMaterial({ color: 0x111820, shininess: 80, specular: 0x223344 });
  const ceilingMat = new THREE.MeshPhongMaterial({ color: 0x0c1018, shininess: 10 });
  const glassMat   = new THREE.MeshPhongMaterial({ color: 0x000d1a, transparent: true, opacity: 0.55, shininess: 300, specular: 0x5599cc });

  // Floor
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(18, 16), floorMat);
  floor.rotation.x = -Math.PI / 2;
  group.add(floor);
  const grid1 = new THREE.GridHelper(18, 36, 0x1a2a3a, 0x0f1a24);
  grid1.position.y = 0.002;
  group.add(grid1);
  for (let z = -6; z <= 6; z += 2) {
    const seam = new THREE.Mesh(new THREE.BoxGeometry(18, 0.015, 0.04), chromeMat);
    seam.position.set(0, 0.008, z);
    group.add(seam);
  }

  // Ceiling
  const ceiling = new THREE.Mesh(new THREE.PlaneGeometry(18, 16), ceilingMat);
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = 6.5;
  group.add(ceiling);
  for (let z = -5; z <= 4; z += 3) {
    const rib = new THREE.Mesh(new THREE.BoxGeometry(18, 0.18, 0.22), steelMat);
    rib.position.set(0, 6.35, z);
    group.add(rib);
    const flangeL = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.32, 0.22), chromeMat);
    const flangeR = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.32, 0.22), chromeMat);
    flangeL.position.set(-8.94, 6.2, z);
    flangeR.position.set(8.94, 6.2, z);
    group.add(flangeL, flangeR);
  }
  const lightStripMat = new THREE.MeshBasicMaterial({ color: 0xaabbcc });
  for (let z = -5; z <= 4; z += 3) {
    const ls = new THREE.Mesh(new THREE.BoxGeometry(14, 0.06, 0.06), lightStripMat);
    ls.position.set(0, 6.28, z);
    group.add(ls);
    const pl = new THREE.PointLight(0x8899aa, 0.4, 8);
    pl.position.set(0, 6.0, z);
    group.add(pl);
  }

  // Back wall — viewport
  const backWall = new THREE.Mesh(new THREE.PlaneGeometry(18, 7), darkSteel);
  backWall.position.set(0, 3.5, -8);
  group.add(backWall);
  const winFrameMat = new THREE.MeshPhongMaterial({ color: 0x1e2830, shininess: 100, specular: 0x445566 });
  const winFrame = new THREE.Mesh(new THREE.BoxGeometry(9.0, 3.2, 0.25), winFrameMat);
  winFrame.position.set(-0.5, 3.8, -7.88);
  group.add(winFrame);
  for (let i = 0; i < 3; i++) {
    const paneW = 2.7, paneH = 2.6;
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(paneW, paneH), glassMat);
    pane.position.set(-2.7 + i * 2.72, 3.8, -7.75);
    group.add(pane);
    const pf = new THREE.Mesh(new THREE.BoxGeometry(paneW + 0.08, paneH + 0.08, 0.14), winFrameMat);
    pf.position.set(-2.7 + i * 2.72, 3.8, -7.82);
    group.add(pf);
  }
  for (let i = 0; i < 2; i++) {
    const div = new THREE.Mesh(new THREE.BoxGeometry(0.14, 2.8, 0.2), chromeMat);
    div.position.set(-1.36 + i * 2.72, 3.8, -7.80);
    group.add(div);
  }

  // Left wall — equipment racks
  const leftWall = new THREE.Mesh(new THREE.PlaneGeometry(16, 7), darkSteel);
  leftWall.rotation.y = Math.PI / 2;
  leftWall.position.set(-8, 3.5, -1);
  group.add(leftWall);
  for (let r = 0; r < 3; r++) {
    const rack = new THREE.Mesh(new THREE.BoxGeometry(0.22, 2.8, 1.8), panelMat);
    rack.rotation.y = Math.PI / 2;
    rack.position.set(-7.89, 1.4, -4 + r * 3.2);
    group.add(rack);
    for (let s = 0; s < 5; s++) {
      const slot = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.28, 1.5), darkSteel);
      slot.rotation.y = Math.PI / 2;
      slot.position.set(-7.80, 0.4 + s * 0.5, -4 + r * 3.2);
      group.add(slot);
      const ledColor = s === 2 ? 0x00ff44 : s === 4 ? 0xff4400 : 0x004488;
      const led = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), new THREE.MeshBasicMaterial({ color: ledColor }));
      led.position.set(-7.77, 0.4 + s * 0.5, -4.6 + r * 3.2);
      group.add(led);
    }
  }

  // Right wall — I-beams
  const rightWall = new THREE.Mesh(new THREE.PlaneGeometry(16, 7), darkSteel);
  rightWall.rotation.y = -Math.PI / 2;
  rightWall.position.set(8, 3.5, -1);
  group.add(rightWall);
  for (let z = -6; z <= 2; z += 4) {
    const web = new THREE.Mesh(new THREE.BoxGeometry(0.08, 6, 0.22), steelMat);
    web.rotation.y = -Math.PI / 2;
    web.position.set(7.7, 3, z);
    group.add(web);
    const flangeTop = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.08, 0.22), chromeMat);
    flangeTop.rotation.y = -Math.PI / 2;
    flangeTop.position.set(7.7, 6.0, z);
    group.add(flangeTop);
    const flangeBot = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.08, 0.22), chromeMat);
    flangeBot.rotation.y = -Math.PI / 2;
    flangeBot.position.set(7.7, 0.04, z);
    group.add(flangeBot);
  }

  // Main console (left)
  const consoleBase = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.9, 1.2), darkSteel);
  consoleBase.position.set(-3.5, 0.45, -2.0);
  group.add(consoleBase);
  const consoleSurface = new THREE.Mesh(new THREE.BoxGeometry(3.5, 0.08, 1.2), steelMat);
  consoleSurface.position.set(-3.5, 0.92, -2.0);
  consoleSurface.rotation.x = 0.18;
  group.add(consoleSurface);
  const screenColors = [0x001a44, 0x001133, 0x002244, 0x001122];
  for (let i = 0; i < 4; i++) {
    const scr = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.52, 0.04), panelMat);
    scr.position.set(-4.8 + i * 0.85, 1.55, -2.3);
    scr.rotation.x = -0.15;
    group.add(scr);
    const face = new THREE.Mesh(new THREE.PlaneGeometry(0.64, 0.44), new THREE.MeshBasicMaterial({ color: screenColors[i] }));
    face.position.set(-4.8 + i * 0.85, 1.55, -2.28);
    face.rotation.x = -0.15;
    group.add(face);
    const sl = new THREE.PointLight(0x002255, 0.25, 1.5);
    sl.position.set(-4.8 + i * 0.85, 1.55, -2.1);
    group.add(sl);
  }
  const kb = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.04, 0.35), steelMat);
  kb.position.set(-3.5, 0.97, -1.6);
  group.add(kb);

  // Secondary console (right)
  const con2 = new THREE.Mesh(new THREE.BoxGeometry(2.0, 1.1, 0.9), darkSteel);
  con2.position.set(5.2, 0.55, -1.5);
  group.add(con2);
  for (let i = 0; i < 2; i++) {
    const s2 = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.5, 0.04), panelMat);
    s2.position.set(4.65 + i * 0.9, 1.38, -1.5);
    s2.rotation.x = -0.2;
    group.add(s2);
    const f2 = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.42), new THREE.MeshBasicMaterial({ color: i ? 0x001a22 : 0x001a00 }));
    f2.position.set(4.65 + i * 0.9, 1.38, -1.48);
    f2.rotation.x = -0.2;
    group.add(f2);
    const sl2 = new THREE.PointLight(i ? 0x002233 : 0x002200, 0.2, 1.2);
    sl2.position.set(4.65 + i * 0.9, 1.4, -1.2);
    group.add(sl2);
  }

  // Overhead conduit runs
  const conduitMat = new THREE.MeshPhongMaterial({ color: 0x2a3540, shininess: 40, specular: 0x445566 });
  for (let c = 0; c < 4; c++) {
    const conduit = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 16, 6), conduitMat);
    conduit.rotation.z = Math.PI / 2;
    conduit.position.set(0, 6.1 + c * 0.1, -3 + c * 0.05);
    group.add(conduit);
  }
  for (let z = -5; z <= 3; z += 2) {
    const dc = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 5, 6), conduitMat);
    dc.position.set(-7.4, 3.5, z);
    group.add(dc);
  }

  // Floor edge trim
  const trimMat = new THREE.MeshPhongMaterial({ color: 0x334455, shininess: 120, specular: 0x667788 });
  const trimF = new THREE.Mesh(new THREE.BoxGeometry(18, 0.06, 0.06), trimMat);
  trimF.position.set(0, 0.03, 8);
  group.add(trimF);
  const trimB = new THREE.Mesh(new THREE.BoxGeometry(18, 0.06, 0.06), trimMat);
  trimB.position.set(0, 0.03, -8);
  group.add(trimB);

  return group;
}

// ── ENDURANCE SPACESHIP ───────────────────────────────────

function buildEndurance() {
  const shipGroup = new THREE.Group();

  const hullMat   = new THREE.MeshPhongMaterial({ color: 0xaab8c2, shininess: 90, specular: 0x445566 });
  const darkMat   = new THREE.MeshPhongMaterial({ color: 0x2a3540, shininess: 40, specular: 0x223344 });
  const chromeMat = new THREE.MeshPhongMaterial({ color: 0x667788, shininess: 200, specular: 0xaabbcc });
  const panelMat  = new THREE.MeshPhongMaterial({ color: 0x8899aa, shininess: 60, specular: 0x334455 });
  const glassMat  = new THREE.MeshPhongMaterial({ color: 0x001a33, transparent: true, opacity: 0.55, shininess: 300, specular: 0x5599cc });
  const thrusterMat = new THREE.MeshPhongMaterial({ color: 0x334455, shininess: 120, specular: 0x8899aa });
  const glowMat   = new THREE.MeshBasicMaterial({ color: 0x003366, transparent: true, opacity: 0.4 });

  const RING_RADIUS = 22;
  const N_MODULES   = 12;

  // Central hub
  const hubBody = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, 5.0, 24), hullMat);
  shipGroup.add(hubBody);

  const hubCapT = new THREE.Mesh(new THREE.CylinderGeometry(3.4, 3.2, 0.4, 24), chromeMat);
  hubCapT.position.y = 2.7;
  shipGroup.add(hubCapT);
  const hubCapB = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.4, 0.4, 24), chromeMat);
  hubCapB.position.y = -2.7;
  shipGroup.add(hubCapB);

  // Docking ports
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2;
    const port = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 1.2, 12), chromeMat);
    port.position.set(Math.cos(a) * 3.8, 0, Math.sin(a) * 3.8);
    port.rotation.z = Math.PI / 2;
    port.rotation.y = a;
    shipGroup.add(port);

    const collar = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.12, 8, 16), darkMat);
    collar.position.set(Math.cos(a) * 4.4, 0, Math.sin(a) * 4.4);
    collar.rotation.y = a;
    shipGroup.add(collar);
  }

  // Antenna tower
  const antBase = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 0.6, 1.5, 8), darkMat);
  antBase.position.y = 3.6;
  shipGroup.add(antBase);
  const antMid = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.4, 2.5, 8), darkMat);
  antMid.position.y = 5.6;
  shipGroup.add(antMid);
  const antTip = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.15, 1.8, 6), chromeMat);
  antTip.position.y = 7.5;
  shipGroup.add(antTip);
  const dish = new THREE.Mesh(new THREE.SphereGeometry(1.4, 16, 8, 0, Math.PI*2, 0, Math.PI/2), panelMat);
  dish.position.y = 8.6;
  shipGroup.add(dish);
  const dSup = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 1.0, 6), chromeMat);
  dSup.position.y = 8.1;
  shipGroup.add(dSup);

  // Spokes
  for (let i = 0; i < N_MODULES; i++) {
    const angle = (i / N_MODULES) * Math.PI * 2;
    const spokeGroup = new THREE.Group();
    spokeGroup.rotation.y = angle;

    const spoke = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, RING_RADIUS - 4.5, 8), darkMat);
    spoke.position.set(0, 0, (RING_RADIUS - 4.5) / 2 + 3.2);
    spoke.rotation.x = Math.PI / 2;
    spokeGroup.add(spoke);

    for (let s = -1; s <= 1; s += 2) {
      const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, RING_RADIUS - 5, 6), chromeMat);
      rod.position.set(s * 0.35, 0, (RING_RADIUS - 5) / 2 + 3.2);
      rod.rotation.x = Math.PI / 2;
      spokeGroup.add(rod);
    }

    const junct = new THREE.Mesh(new THREE.SphereGeometry(0.45, 8, 8), chromeMat);
    junct.position.set(0, 0, 3.5);
    spokeGroup.add(junct);

    shipGroup.add(spokeGroup);
  }

  // Habitat modules
  for (let i = 0; i < N_MODULES; i++) {
    const angle = (i / N_MODULES) * Math.PI * 2;
    const modGroup = new THREE.Group();
    modGroup.rotation.y = angle;

    const mx = 0, my = 0, mz = RING_RADIUS;

    const box = new THREE.Mesh(new THREE.BoxGeometry(3.8, 2.6, 4.2), hullMat.clone());
    box.position.set(mx, my, mz);
    modGroup.add(box);

    const face = new THREE.Mesh(new THREE.BoxGeometry(3.7, 2.4, 0.1), panelMat);
    face.position.set(mx, my, mz + 2.15);
    modGroup.add(face);

    for (let w = -1; w <= 1; w += 2) {
      const win = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.65, 0.08), glassMat);
      win.position.set(mx + w * 1.1, my + 0.3, mz + 2.22);
      modGroup.add(win);

      const wf = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.78, 0.05), chromeMat);
      wf.position.set(mx + w * 1.1, my + 0.3, mz + 2.20);
      modGroup.add(wf);

      const wg = new THREE.Mesh(new THREE.PlaneGeometry(0.85, 0.60), glowMat);
      wg.position.set(mx + w * 1.1, my + 0.3, mz + 2.24);
      modGroup.add(wg);
    }

    const capL = new THREE.Mesh(new THREE.BoxGeometry(3.8, 2.6, 0.18), chromeMat);
    capL.position.set(mx, my, mz - 2.19);
    modGroup.add(capL);
    const capR = new THREE.Mesh(new THREE.BoxGeometry(3.8, 2.6, 0.18), chromeMat);
    capR.position.set(mx, my, mz + 2.19);
    modGroup.add(capR);

    const eqBay = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.7, 3.6), darkMat);
    eqBay.position.set(mx, my - 1.65, mz);
    modGroup.add(eqBay);

    for (let s = -1; s <= 1; s += 2) {
      const rad = new THREE.Mesh(new THREE.BoxGeometry(0.05, 2.0, 3.0), panelMat);
      rad.position.set(mx + s * 2.1, my, mz);
      modGroup.add(rad);

      for (let r = 0; r < 4; r++) {
        const rib = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.06, 3.0), chromeMat);
        rib.position.set(mx + s * 2.12, my - 0.7 + r * 0.5, mz);
        modGroup.add(rib);
      }
    }

    shipGroup.add(modGroup);
  }

  // Propulsion section
  const ENGINE_Y = -5.5;
  for (let i = 0; i < 3; i++) {
    const ea = (i / 3) * Math.PI * 2;
    const ex = Math.cos(ea) * 5.5;
    const ez = Math.sin(ea) * 5.5;

    const pod = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 1.3, 6.0, 12), darkMat);
    pod.position.set(ex, ENGINE_Y, ez);
    shipGroup.add(pod);

    const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(1.3, 0.6, 1.2, 12), thrusterMat);
    nozzle.position.set(ex, ENGINE_Y - 3.7, ez);
    shipGroup.add(nozzle);

    const egMat = new THREE.MeshBasicMaterial({ color: 0x003366, transparent: true, opacity: 0.35 });
    const eGlow = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.6, 0.3, 12), egMat);
    eGlow.position.set(ex, ENGINE_Y - 4.5, ez);
    shipGroup.add(eGlow);

    const conn = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 4.5, 6), darkMat);
    conn.position.set(ex * 0.5, ENGINE_Y, ez * 0.5);
    conn.rotation.z = Math.PI / 2;
    conn.rotation.y = ea;
    shipGroup.add(conn);
  }

  // Solar arrays
  for (let s = -1; s <= 1; s += 2) {
    const saSupport = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 8, 6), chromeMat);
    saSupport.position.set(s * 5.5, 4.5, 0);
    saSupport.rotation.z = Math.PI / 2;
    shipGroup.add(saSupport);

    const solarPanel = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.06, 2.8), panelMat);
    solarPanel.position.set(s * 9.5, 4.5, 0);
    shipGroup.add(solarPanel);

    for (let r = 0; r < 5; r++) {
      const line = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.04, 0.04), darkMat);
      line.position.set(s * 9.5, 4.52, -1.1 + r * 0.56);
      shipGroup.add(line);
    }
    for (let c = 0; c < 9; c++) {
      const cline = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 2.8), darkMat);
      cline.position.set(s * 9.5 - 3.2 + c * 0.72, 4.52, 0);
      shipGroup.add(cline);
    }
  }

  return shipGroup;
}

// ── STARFIELD ──────────────────────────────────────────────

function buildStarfield() {
  const sg = new THREE.BufferGeometry();
  const count = 5000;
  const sp = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    const r     = 200 + Math.random() * 300;
    sp[i*3]   = r * Math.sin(phi) * Math.cos(theta);
    sp[i*3+1] = r * Math.sin(phi) * Math.sin(theta);
    sp[i*3+2] = r * Math.cos(phi);
  }
  sg.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  return new THREE.Points(sg, new THREE.PointsMaterial({ color: 0xddeeff, size: 0.7 }));
}

// ── ASTRONAUT LOADING ───────────────────────────────────

function loadAstronautGLB() {
  if (!window.GLTFLoader) {
    buildFallbackAstronaut();
    return;
  }
  const loader = new window.GLTFLoader();
  loader.load(
    '/static/models/Astronaut.glb',
    function(gltf) {
      while (astronaut.children.length > 0) {
        astronaut.remove(astronaut.children[0]);
      }
      const model = gltf.scene;
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      const targetHeight = 2.2;
      const scaleFactor = targetHeight / size.y;
      model.scale.setScalar(scaleFactor);
      box.setFromObject(model);
      const newCenter = box.getCenter(new THREE.Vector3());
      const newMin = box.min;
      model.position.x = -newCenter.x;
      model.position.y = -newMin.y;
      model.position.z = -newCenter.z;
      astronaut.add(model);
      glbLoaded = true;
    },
    null,
    function(error) {
      console.warn('Astronaut.glb load failed, using fallback:', error);
      buildFallbackAstronaut();
    }
  );
}

function buildFallbackAstronaut() {
  const mat = new THREE.MeshPhongMaterial({ color: 0xcccccc, shininess: 35 });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.38, 1.3, 16), mat);
  body.position.y = 1.5;
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.38, 16, 16),
    new THREE.MeshPhongMaterial({ color: 0xeeeeee, shininess: 80 }));
  head.position.y = 2.6;
  astronaut.add(body, head);
  bodyMesh = body;
  glbLoaded = true;
}

// ── CAMERA CONTROL ──────────────────────────────────────

function setCamMode(mode) {
  camMode = mode;
  if (mode === 'ship') {
    cam3.up.set(0, 1, 0);
    camState.radius = 110; camState.theta = 0.4; camState.phi = 0.1;
    camState.targetY = 0; camState.autoRotate = true;
    document.getElementById('btn-cam-ship').style.borderColor = 'var(--cyan)';
    document.getElementById('btn-cam-ship').style.color = 'var(--cyan)';
    document.getElementById('btn-cam-astro').style.borderColor = '';
    document.getElementById('btn-cam-astro').style.color = '';
    const btn = document.getElementById('btn-set-astro-cam');
    if (btn) btn.style.display = 'none';
  } else {
    cam3.up.set(0, 1, 0);
    camState.radius  = astroCamDefault.radius;
    camState.theta   = astroCamDefault.theta;
    camState.phi     = astroCamDefault.phi;
    camState.targetY = astroCamDefault.targetY;
    camState.autoRotate = false;
    if (astroLookTarget) astroLookTarget.set(astroCamDefault.lookX, astroCamDefault.lookY, astroCamDefault.lookZ);
    document.getElementById('btn-cam-astro').style.borderColor = 'var(--cyan)';
    document.getElementById('btn-cam-astro').style.color = 'var(--cyan)';
    document.getElementById('btn-cam-ship').style.borderColor = '';
    document.getElementById('btn-cam-ship').style.color = '';
    const btn = document.getElementById('btn-set-astro-cam');
    if (btn) btn.style.display = '';
  }
}

function setAstroCamAsDefault() {
  astroCamDefault.radius  = camState.radius;
  astroCamDefault.theta   = camState.theta;
  astroCamDefault.phi     = camState.phi;
  astroCamDefault.targetY = camState.targetY;

  const cxEl = document.getElementById("db-cx");
  if (cxEl) {
    astroCamDefault.lookX = parseFloat(cxEl.value) / 10;
    astroCamDefault.lookY = parseFloat(document.getElementById("db-cy").value) / 10;
    astroCamDefault.lookZ = parseFloat(document.getElementById("db-cz").value) / 10;
  } else if (astroLookTarget) {
    astroCamDefault.lookX = astroLookTarget.x;
    astroCamDefault.lookY = astroLookTarget.y;
    astroCamDefault.lookZ = astroLookTarget.z;
  }

  if (cxEl) {
    document.getElementById("db-cth").value = Math.round(astroCamDefault.theta * 100);
    document.getElementById("db-cph").value = Math.round(astroCamDefault.phi * 100);
    document.getElementById("db-cr").value  = Math.round(astroCamDefault.radius * 10);
    document.getElementById("db-cthv").textContent = (astroCamDefault.theta * 180 / Math.PI).toFixed(0) + "°";
    document.getElementById("db-cphv").textContent = (astroCamDefault.phi  * 180 / Math.PI).toFixed(0) + "°";
    document.getElementById("db-crv").textContent  = astroCamDefault.radius.toFixed(1);
  }

  const btn = document.getElementById('btn-set-astro-cam');
  if (!btn) return;
  const orig = btn.textContent;
  btn.textContent = '✓ POSITION SAVED';
  btn.style.borderColor = 'var(--green)';
  btn.style.color = 'var(--green)';
  setTimeout(() => {
    btn.textContent = orig;
    btn.style.borderColor = 'var(--amber)';
    btn.style.color = 'var(--amber)';
  }, 1400);
}

// ── INITIALIZATION ──────────────────────────────────────

function initThree() {
  const cv = document.getElementById('three-canvas');
  const tw = document.getElementById('panel-twin');
  const W = Math.min(tw.clientWidth - 22, 520);
  const H = Math.round(W * 0.7);
  cv.width = W; cv.height = H;

  renderer3 = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
  renderer3.setSize(W, H);
  renderer3.setClearColor(0x020810, 1);
  renderer3.shadowMap.enabled = true;

  scene3 = new THREE.Scene();

  cam3 = new THREE.PerspectiveCamera(52, W / H, 0.1, 2000);
  cam3.position.set(0, 12, 110);
  cam3.lookAt(0, 0, 0);

  // Space ambient
  aimb = new THREE.AmbientLight(0x111833, 1.0);
  scene3.add(aimb);
  ambientBaseColor = aimb.color.clone();

  // Sun light
  const sunLight = new THREE.DirectionalLight(0xfff8ee, 2.8);
  sunLight.position.set(80, 40, 60);
  scene3.add(sunLight);

  // Earth fill
  const earthFill = new THREE.DirectionalLight(0x2244aa, 0.5);
  earthFill.position.set(-60, -20, -40);
  scene3.add(earthFill);

  // Starfield
  scene3.add(buildStarfield());

  // Load Endurance GLB or fallback
  const endLoader = new window.GLTFLoader()
  endLoader.load(
    '/static/models/Endurance.glb',
    function(gltf) {
      enduranceRing = gltf.scene;
      const box = new THREE.Box3().setFromObject(enduranceRing);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const targetSize = 90;
      enduranceRing.scale.setScalar(targetSize / maxDim);
      box.setFromObject(enduranceRing);
      const centre = box.getCenter(new THREE.Vector3());
      enduranceRing.position.sub(centre);
      scene3.add(enduranceRing);
      if (roomContainer) { scene3.remove(roomContainer); enduranceRing.add(roomContainer); }
    },
    null,
    function(err) {
      console.warn("Endurance.glb not found, using procedural model");
      enduranceRing = buildEndurance();
      enduranceRing.scale.set(2.0, 2.0, 2.0);
      scene3.add(enduranceRing);
      if (roomContainer) { scene3.remove(roomContainer); enduranceRing.add(roomContainer); }
    }
  );

  // Habitat light
  habitatLight = new THREE.PointLight(0xaabbdd, 2.5, 18);
  habitatLight.position.set(0, 5, 0);
  habitatBaseColor = habitatLight.color.clone();

  // Load Black Hole
  const bhLoader = new window.GLTFLoader()
  bhLoader.load('/static/models/BlackHole.glb', function(gltf) {
      const bh = gltf.scene;
      bh.scale.setScalar(3);
      bh.position.set(1000, -100, -1000);
      scene3.add(bh);
  });

  // Room container
  roomContainer = new THREE.Group();
  roomContainer.position.set(7.80, 5.60, 7.50);
  roomContainer.scale.setScalar(0.20);

  const room = buildSpaceshipInterior();
  roomContainer.add(room);

  // Astronaut
  astronaut = new THREE.Group();
  astronaut.position.set(0, 0.40, 0.10);
  roomContainer.add(astronaut);
  loadAstronautGLB();

  // TARS
  tarsGroup = buildTARS();
  tarsGroup.position.set(2.90, 0.00, 0.10);
  tarsGroup.rotation.y = -0.65;
  roomContainer.add(tarsGroup);

  // Risk ring
  riskRingMat = new THREE.MeshBasicMaterial({ color: 0x00e87a, transparent: true, opacity: 0.55 });
  riskRing = new THREE.Mesh(new THREE.TorusGeometry(1.1, 0.04, 8, 80), riskRingMat);
  riskRing.position.set(0, 1.5, 0);
  roomContainer.add(riskRing);

  // Habitat light
  roomContainer.add(habitatLight);

  scene3.add(roomContainer);

  astroLookTarget = new THREE.Vector3(astroCamDefault.lookX, astroCamDefault.lookY, astroCamDefault.lookZ);

  threeOK = true;

  // Camera controls
  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 1.10 : 0.91;
    camState.radius = Math.max(camState.minRadius, Math.min(camState.maxRadius, camState.radius * delta));
  }, { passive: false });

  cv.addEventListener('mousedown', (e) => {
    markUserInteraction();
    camState.isDragging = true;
    camState.autoRotate = false;
    camState.lastX = e.clientX;
    camState.lastY = e.clientY;
  });
  window.addEventListener('mousemove', (e) => {
    if (!camState.isDragging) return;
    const dx = e.clientX - camState.lastX;
    const dy = e.clientY - camState.lastY;
    camState.theta -= dx * 0.008;
    camState.phi = Math.max(-Math.PI/2.2, Math.min(Math.PI/2.2, camState.phi - dy * 0.006));
    camState.lastX = e.clientX;
    camState.lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => { camState.isDragging = false; });

  cv.addEventListener('touchstart', (e) => {
    markUserInteraction();
    if (e.touches.length === 1) {
      camState.isDragging = true;
      camState.autoRotate = false;
      camState.lastX = e.touches[0].clientX;
      camState.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2) {
      camState.isDragging = false;
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      camState.lastPinchDist = Math.sqrt(dx*dx + dy*dy);
    }
  }, { passive: true });
  cv.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (e.touches.length === 1 && camState.isDragging) {
      const dx = e.touches[0].clientX - camState.lastX;
      const dy = e.touches[0].clientY - camState.lastY;
      camState.theta -= dx * 0.008;
      camState.phi = Math.max(-Math.PI/2.2, Math.min(Math.PI/2.2, camState.phi - dy * 0.006));
      camState.lastX = e.touches[0].clientX;
      camState.lastY = e.touches[0].clientY;
    } else if (e.touches.length === 2 && camState.lastPinchDist !== null) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.sqrt(dx*dx + dy*dy);
      const scale = camState.lastPinchDist / dist;
      camState.radius = Math.max(camState.minRadius, Math.min(camState.maxRadius, camState.radius * scale));
      camState.lastPinchDist = dist;
    }
  }, { passive: false });
  cv.addEventListener('touchend', () => { camState.isDragging = false; camState.lastPinchDist = null; });

  cv.addEventListener('dblclick', () => { setCamMode(camMode); });

  initInspectHandlers();

  function syncCamSliders() {
    if (typeof dbOpen === 'undefined' || !dbOpen) return;
    const thEl = document.getElementById("db-cth");
    const phEl = document.getElementById("db-cph");
    const crEl = document.getElementById("db-cr");
    if (thEl) { thEl.value = Math.round(camState.theta * 100); document.getElementById("db-cthv").textContent = (camState.theta * 180 / Math.PI).toFixed(0) + "°"; }
    if (phEl) { phEl.value = Math.round(camState.phi * 100);   document.getElementById("db-cphv").textContent = (camState.phi * 180 / Math.PI).toFixed(0) + "°"; }
    if (crEl) { crEl.value = Math.round(camState.radius * 10); document.getElementById("db-crv").textContent = camState.radius.toFixed(1); }
  }
  window.addEventListener('mousemove', () => { if (camState.isDragging) syncCamSliders(); });
  cv.addEventListener('touchmove', () => { syncCamSliders(); }, { passive: true });
  cv.addEventListener('wheel', () => { syncCamSliders(); }, { passive: true });

  loopThree();
}

// ── RENDER LOOP ─────────────────────────────────────────

function loopThree(ts = 0) {
  requestAnimationFrame(loopThree);
  if (!threeOK) return;

  const dt = Math.min((ts - _t3last) / 1000, 0.05);
  _t3last = ts;
  const t = ts * 0.001;

  let lookTargetX = 0, lookTargetY = camState.targetY, lookTargetZ = 0;

  if (camState.autoRotate) {
    camState.theta += 0.003;
  }

  if (camMode === 'astro' && astronaut && astroLookTarget && roomContainer) {
    const roomQuat = new THREE.Quaternion();
    roomContainer.getWorldQuaternion(roomQuat);

    const aw = new THREE.Vector3();
    astronaut.getWorldPosition(aw);
    const rotatedLook = astroLookTarget.clone().applyQuaternion(roomQuat);
    lookTargetX = aw.x + rotatedLook.x;
    lookTargetY = aw.y + rotatedLook.y;
    lookTargetZ = aw.z + rotatedLook.z;

    const localOffset = new THREE.Vector3(
      camState.radius * Math.cos(camState.phi) * Math.sin(camState.theta),
      camState.radius * Math.sin(camState.phi),
      camState.radius * Math.cos(camState.phi) * Math.cos(camState.theta)
    ).applyQuaternion(roomQuat);

    cam3.position.set(lookTargetX + localOffset.x, lookTargetY + localOffset.y, lookTargetZ + localOffset.z);
  } else {
    const cx = lookTargetX + camState.radius * Math.cos(camState.phi) * Math.sin(camState.theta);
    const cy = lookTargetY + camState.radius * Math.sin(camState.phi);
    const cz = lookTargetZ + camState.radius * Math.cos(camState.phi) * Math.cos(camState.theta);
    cam3.position.set(cx, cy, cz);
  }
  cam3.lookAt(lookTargetX, lookTargetY, lookTargetZ);

  // Endurance ring rotation
  if (enduranceRing && camMode === 'ship') {
    enduranceRing.rotation.y += 0.0018;
  }

  // TARS animation
  if (tarsGroup) {
    tarsGroup.rotation.y = -0.65 + Math.sin(t * 0.3) * 0.06;
    const scan = tarsGroup.getObjectByName('tars_scan');
    if (scan) scan.position.y = 0.15 + Math.sin(t * 1.8) * 0.25;
    if (tarsScreenMat) {
      tarsLight.intensity = (0.7 + Math.sin(t * 2.1) * 0.3) * 0.9;
    }
  }

  if (!_simState) {
    renderer3.render(scene3, cam3);
    return;
  }

  const { fatigue, hr, smsSev, stress, risk, phase, fatigueIndex, motionSeverity } = _simState;

  // Astronaut phase animations
  if (phase === 'SLEEP') {
    astronaut.rotation.x += ((-Math.PI / 2) - astronaut.rotation.x) * 0.04;
    astronaut.position.y += (1 - astronaut.position.y) * 0.04;
    astronaut.position.z += (0.10 - astronaut.position.z) * 0.04;
    aimb.intensity += (0.7 - aimb.intensity) * 0.03;
  } else {
    astronaut.rotation.x += (0 - astronaut.rotation.x) * 0.04;
    astronaut.position.y += (0.40 - astronaut.position.y) * 0.04;
    astronaut.position.z += (0.10 - astronaut.position.z) * 0.04;
    aimb.intensity += (1.4 - aimb.intensity) * 0.03;
  }

  const fatigueNorm = Math.max(0, Math.min(1, (Number(fatigueIndex) || fatigue * 10) / 10));
  const speedFactor = 1 - 0.5 * fatigueNorm;
  const swaySpeed = (1.2 - fatigue * 0.8) * speedFactor;
  astronaut.rotation.z = Math.sin(t * swaySpeed) * (fatigue * 0.06 + 0.01);

  if (smsSev > 0.1) {
    astronaut.rotation.z += Math.sin(t * (4.5 * speedFactor)) * smsSev * 0.09;
    astronaut.position.x = Math.sin(t * (3.8 * speedFactor)) * smsSev * 0.06;
  } else {
    astronaut.position.x *= 0.95;
  }

  const motSev = Number(motionSeverity ?? (smsSev * 5)) || 0;
  if (motSev > 2.0) {
    const jit = Math.min(0.05, 0.005 + (motSev - 2.0) * 0.008);
    tremorX = tremorX * 0.78 + (Math.random() - 0.5) * jit;
    tremorZ = tremorZ * 0.78 + (Math.random() - 0.5) * jit;
    astronaut.position.x += tremorX;
    astronaut.position.z += tremorZ;
  } else {
    tremorX *= 0.8;
    tremorZ *= 0.8;
  }

  const hrRad = (hr / 60) * 2 * Math.PI;
  const pulse = 1 + Math.sin(t * hrRad) * 0.012;
  astronaut.scale.x = pulse;
  astronaut.scale.z = pulse;

  if (tarsScreenMat) {
    if (stress > 0.7) tarsScreenMat.color.setHex(0xff4040);
    else if (stress > 0.4) tarsScreenMat.color.setHex(0xffaa00);
    else tarsScreenMat.color.setHex(0x00d4ff);
  }

  if (risk > 0.7) highRiskLighting = true;
  else if (risk < 0.5) highRiskLighting = false;
  if (aimb && ambientBaseColor) {
    const targetA = highRiskLighting
      ? ambientBaseColor.clone().lerp(new THREE.Color(0xff4040), 0.25 + 0.1 * Math.sin(t * 4.2))
      : ambientBaseColor;
    aimb.color.lerp(targetA, 0.1);
  }
  if (habitatLight && habitatBaseColor) {
    const pulse = 0.2 + 0.2 * Math.sin(t * 5.5);
    const targetH = highRiskLighting
      ? habitatBaseColor.clone().lerp(new THREE.Color(0xff4040), pulse)
      : habitatBaseColor;
    habitatLight.color.lerp(targetH, 0.14);
  }

  // Risk ring
  const rc = risk < 0.4 ? new THREE.Color(0x00e87a) : risk < 0.7 ? new THREE.Color(0xffaa00) : new THREE.Color(0xff4040);
  riskRingMat.color.lerp(rc, 0.08);
  riskRing.position.x = astronaut.position.x;
  riskRing.position.z = astronaut.position.z;
  riskRing.rotation.z = t * 0.8;

  renderer3.render(scene3, cam3);
}