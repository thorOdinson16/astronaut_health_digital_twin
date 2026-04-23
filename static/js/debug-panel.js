// static/js/debug-panel.js
// ════════════════════════════════════════════════════════
// DEBUG POSITIONING TOOL
// ════════════════════════════════════════════════════════

let dbAxes = null, dbOpen = false;

function toggleDebug() {
  dbOpen = !dbOpen;
  document.getElementById("dbpanel").style.display = dbOpen ? "block" : "none";
  document.getElementById("dbtn").classList.toggle("active", dbOpen);
  
  if (dbOpen) {
    setCamMode("astro");
    if (!dbAxes && scene3) {
      dbAxes = new THREE.AxesHelper(12);
      scene3.add(dbAxes);
    }
    if (dbAxes) dbAxes.visible = true;
    
    // Sync sliders FROM current state — never overwrite with defaults
    if (roomContainer) {
      document.getElementById("db-rx").value = Math.round(roomContainer.position.x * 10);
      document.getElementById("db-rY").value = Math.round(roomContainer.position.y * 10);
      document.getElementById("db-rz").value = Math.round(roomContainer.position.z * 10);
      document.getElementById("db-rs").value = Math.round(roomContainer.scale.x * 10);
    }
    if (astronaut) {
      document.getElementById("db-x").value = Math.round(astronaut.position.x * 10);
      document.getElementById("db-y").value = Math.round(astronaut.position.y * 10);
      document.getElementById("db-z").value = Math.round(astronaut.position.z * 10);
      document.getElementById("db-s").value = Math.round(astronaut.scale.x * 10);
      document.getElementById("db-ry").value = Math.round(astronaut.rotation.y * 180 / Math.PI);
    }
    
    // Sync TARS offset
    if (tarsGroup) {
      document.getElementById("db-tx").value = Math.round((tarsGroup.position.x - astronaut.position.x) * 10);
    }
    
    // Sync all camera sliders from live state
    document.getElementById("db-cth").value = Math.round(camState.theta * 100);
    document.getElementById("db-cph").value = Math.round(camState.phi * 100);
    document.getElementById("db-cr").value  = Math.round(camState.radius * 10);
    
    if (astroLookTarget) {
      document.getElementById("db-cx").value = Math.round(astroLookTarget.x * 10);
      document.getElementById("db-cy").value = Math.round(astroLookTarget.y * 10);
      document.getElementById("db-cz").value = Math.round(astroLookTarget.z * 10);
    }
    
    // Update display values
    applyPos();
  } else {
    if (dbAxes) dbAxes.visible = false;
  }
}

// Coarse nudge buttons (each click = 1 unit)
function nudge(axis, dir) {
  const el = document.getElementById("db-" + axis);
  if (!el) return;
  el.value = +el.value + dir;
  applyPos();
}

function applyPos() {
  if (!astronaut || !tarsGroup || !roomContainer) return;

  // Room controls
  const rx  = document.getElementById("db-rx")?.value / 10 || 0;
  const rY  = document.getElementById("db-rY")?.value / 10 || 0;
  const rz  = document.getElementById("db-rz")?.value / 10 || 0;
  const rs  = document.getElementById("db-rs")?.value / 10 || 1;

  roomContainer.position.set(rx, rY, rz);
  roomContainer.scale.setScalar(rs);

  // Astronaut controls (relative to room)
  const x  = document.getElementById("db-x")?.value / 10 || 0;
  const y  = document.getElementById("db-y")?.value / 10 || 0;
  const z  = document.getElementById("db-z")?.value / 10 || 0;
  const s  = document.getElementById("db-s")?.value / 10 || 1;
  const ry = (document.getElementById("db-ry")?.value || 0) * Math.PI / 180;
  const tx = document.getElementById("db-tx")?.value / 10 || 0;

  astronaut.position.set(x, y, z);
  astronaut.scale.setScalar(s);
  astronaut.rotation.y = ry;

  tarsGroup.position.set(x + tx, y, z);
  tarsGroup.scale.setScalar(s * 0.85);
  tarsGroup.rotation.y = ry + 0.4;

  if (riskRing) riskRing.position.set(x, y + 1.5 * s, z);

  // Camera look target
  const cx = document.getElementById("db-cx")?.value / 10 || 0;
  const cy = document.getElementById("db-cy")?.value / 10 || 0;
  const cz = document.getElementById("db-cz")?.value / 10 || 0;
  const cr = document.getElementById("db-cr")?.value / 10 || 5;

  if (astroLookTarget) astroLookTarget.set(cx, cy, cz);
  if (camMode === 'astro') {
    camState.radius = cr;
    camState.theta  = (document.getElementById("db-cth")?.value || 0) / 100;
    camState.phi    = Math.max(-Math.PI/2.2, Math.min(Math.PI/2.2,
                        (document.getElementById("db-cph")?.value || 0) / 100));
    camState.autoRotate = false;
  }

  // Axes at astronaut world position
  if (dbAxes) {
    const wp = new THREE.Vector3();
    astronaut.getWorldPosition(wp);
    dbAxes.position.copy(wp);
  }

  // Update display values
  updateDisplayValues(rx, rY, rz, rs, x, y, z, s, ry, tx);
}

function updateDisplayValues(rx, rY, rz, rs, x, y, z, s, ry, tx) {
  const updates = {
    "db-rxv": (+rx).toFixed(1),
    "db-rYv": (+rY).toFixed(1),
    "db-rzv": (+rz).toFixed(1),
    "db-rsv": (+rs).toFixed(1) + "×",
    "db-xv": (+x).toFixed(1),
    "db-yv": (+y).toFixed(1),
    "db-zv": (+z).toFixed(1),
    "db-sv": (+s).toFixed(1) + "×",
    "db-ryv": (document.getElementById("db-ry")?.value || 0) + "°",
    "db-txv": "+" + (+tx).toFixed(1),
    "db-cxv": ((document.getElementById("db-cx")?.value || 0) / 10).toFixed(1),
    "db-cyv": ((document.getElementById("db-cy")?.value || 0) / 10).toFixed(1),
    "db-czv": ((document.getElementById("db-cz")?.value || 0) / 10).toFixed(1),
    "db-crv": ((document.getElementById("db-cr")?.value || 0) / 10).toFixed(1),
    "db-cthv": (camState.theta * 180 / Math.PI).toFixed(0) + "°",
    "db-cphv": (camState.phi * 180 / Math.PI).toFixed(0) + "°",
  };

  Object.entries(updates).forEach(([id, val]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  });

  // Update copy-able code
  const code = `roomContainer.position.set(${(+rx).toFixed(2)}, ${(+rY).toFixed(2)}, ${(+rz).toFixed(2)});\n`
    + `roomContainer.scale.setScalar(${(+rs).toFixed(2)});\n`
    + `astronaut.position.set(${(+x).toFixed(2)}, ${(+y).toFixed(2)}, ${(+z).toFixed(2)});\n`
    + `tarsGroup.position.set(${(+(x+tx)).toFixed(2)}, ${(+y).toFixed(2)}, ${(+z).toFixed(2)});`;
  
  const codeEl = document.getElementById("db-code");
  if (codeEl) codeEl.textContent = code;
}

function copyCode() {
  const code = document.getElementById("db-code")?.textContent;
  if (!code) return;
  navigator.clipboard.writeText(code).then(() => {
    const el = document.getElementById("db-code");
    if (!el) return;
    const orig = el.style.borderColor;
    el.style.borderColor = "#00d4ff";
    setTimeout(() => el.style.borderColor = orig, 800);
  });
}

// Initialize debug panel button
document.addEventListener('DOMContentLoaded', () => {
  const dbtn = document.getElementById('dbtn');
  if (dbtn) {
    dbtn.addEventListener('click', toggleDebug);
  }
});

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && e.key === 'D') {
    e.preventDefault();
    toggleDebug();
  }
});