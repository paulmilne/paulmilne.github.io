// CGI view of Apollo 11 for the Part 1 player.
// Everything is built from primitives at real dimensions (metres); no external models.
// World frame: geocentric, mean equator of date (z = north pole), floating origin at the camera target.

const EVENTS = {
  sicSep: 162.3,          // S-IC/S-II separation (ABTN)
  skirtSep: 193,          // "We've got skirt SEP" (air-to-ground transcript)
  tower: 197,             // "Tower's gone" (transcript)
  siiSep: 549.0,          // S-II/S-IVB separation (ABTN)
  eoi: 709.33,            // S-IVB first cutoff + 10 s (ABTN)
  tliIgn: 9856.2, tliCut: 10203.03,
  csmSep: 11723.0,        // CSM separated from S-IVB (ABTN)
  dock: 12243.7,          // CSM docked with LM/S-IVB (ABTN)
  ejection: 15420,        // LM ejection, about 4:17 GET (crew "ready for LM ejection" 4:16:38)
  evasive: 16801,         // evasive SPS burn GETI 4:40:01 (PAD read up in transcript), 3 s
};
export {EVENTS};

const OMEGA = 7.2921158553e-5;

export function createScene(THREE, {data, land, d3, width, height}) {
  const W = width, H = height;
  const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H);
  renderer.outputEncoding = THREE.sRGBEncoding;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02040a);
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.5, 2e9);

  // Light: the Sun (direction updated per frame) and a little fill.
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  scene.add(sun, sun.target);
  scene.add(new THREE.AmbientLight(0x8899aa, 0.32));

  // Stars.
  let stars;
  {
    const g = new THREE.BufferGeometry(), n = 2500, p = new Float32Array(n * 3);
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) {
      const z = 2 * rnd() - 1, a = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
      p.set([1e9 * r * Math.cos(a), 1e9 * r * Math.sin(a), 1e9 * z], 3 * i);
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    stars = new THREE.Points(g, new THREE.PointsMaterial({color: 0xffffff, size: 1.2, sizeAttenuation: false}));
    scene.add(stars);
  }

  // Earth: lat/long sphere with a texture drawn from the same land outlines as the globe panel.
  const RE = 6378166, FLAT = 1 / 298.3;          // Fischer ellipsoid, as in the 1969 tables
  const earth = new THREE.Group();
  {
    const tw = 2048, th = 1024, cv = document.createElement("canvas");
    cv.width = tw; cv.height = th;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#123a63"; ctx.fillRect(0, 0, tw, th);
    const proj = d3.geoEquirectangular().scale(tw / (2 * Math.PI)).translate([tw / 2, th / 2]);
    const path = d3.geoPath(proj, ctx);
    ctx.beginPath(); path(land); ctx.fillStyle = "#4a6b3d"; ctx.fill();
    ctx.beginPath(); path(d3.geoGraticule10()); ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.stroke();
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    tex.anisotropy = 8;
    const nLat = 90, nLon = 180, pos = [], uv = [], idx = [];
    for (let i = 0; i <= nLat; i++) {
      const lat = -Math.PI / 2 + Math.PI * i / nLat;
      for (let j = 0; j <= nLon; j++) {
        const lon = -Math.PI + 2 * Math.PI * j / nLon;
        pos.push(RE * Math.cos(lat) * Math.cos(lon), RE * Math.cos(lat) * Math.sin(lon), RE * Math.sin(lat));
        uv.push(j / nLon, i / nLat);
      }
    }
    for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
      const a = i * (nLon + 1) + j, b = a + nLon + 1;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx); g.computeVertexNormals();
    earth.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({map: tex, roughness: 0.9, metalness: 0})));
    const atm = new THREE.Mesh(new THREE.SphereGeometry(RE * 1.015, 96, 48),
      new THREE.MeshBasicMaterial({color: 0x5aa0ff, transparent: true, opacity: 0.12, side: THREE.BackSide}));
    earth.add(atm);
  }
  earth.scale.set(1, 1, 1 - FLAT);
  // Launch Complex 39A: pad and launch umbilical tower, fixed to the Earth.
  const padLat = 28.4470 * Math.PI / 180, padLon = -80.6041 * Math.PI / 180;
  const padR = 1 / Math.hypot(Math.cos(padLat) / RE, Math.sin(padLat) / (RE * (1 - FLAT)));
  {
    const up = new THREE.Vector3(Math.cos(padLat) * Math.cos(padLon), Math.cos(padLat) * Math.sin(padLon), Math.sin(padLat));
    const site = new THREE.Group();
    const padMesh = new THREE.Mesh(new THREE.CylinderGeometry(70, 80, 8, 32), new THREE.MeshStandardMaterial({color: 0x8a8578, roughness: 1}));
    padMesh.position.y = -4; site.add(padMesh);
    const lut = new THREE.Mesh(new THREE.BoxGeometry(12, 120, 12), new THREE.MeshStandardMaterial({color: 0xb5452b, roughness: 0.8, wireframe: true}));
    lut.position.set(0, 60, -19); site.add(lut);
    site.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);
    site.position.copy(up.clone().multiplyScalar(padR)).multiply(new THREE.Vector3(1, 1, 1 / (1 - FLAT)));
    site.scale.set(1, 1, 1);
    earth.add(site);
  }
  scene.add(earth);
  const moon = new THREE.Mesh(new THREE.SphereGeometry(1737400, 48, 24),
    new THREE.MeshStandardMaterial({color: 0x9a9a96, roughness: 1}));
  scene.add(moon);

  // ---- The vehicle. Local frame: +y forward (nose), y = 0 at the S-IC base. ----
  const white = new THREE.MeshStandardMaterial({color: 0xf2f2ee, roughness: 0.55});
  const black = new THREE.MeshStandardMaterial({color: 0x1a1a1a, roughness: 0.6});
  const grey = new THREE.MeshStandardMaterial({color: 0x9aa0a6, roughness: 0.5, metalness: 0.3});
  const silver = new THREE.MeshStandardMaterial({color: 0xd8dade, roughness: 0.25, metalness: 0.85});
  const engineMat = new THREE.MeshStandardMaterial({color: 0x2b2b2b, roughness: 0.4, metalness: 0.6, side: THREE.DoubleSide});
  const gold = new THREE.MeshStandardMaterial({color: 0xc9a227, roughness: 0.35, metalness: 0.7});

  const stripeTex = (pattern) => {
    const cv = document.createElement("canvas"); cv.width = 256; cv.height = 256;
    const c = cv.getContext("2d");
    pattern(c); const t = new THREE.CanvasTexture(cv); t.encoding = THREE.sRGBEncoding; return t;
  };
  // S-IC roll pattern: alternating black and white quarters on the lower tank, black intertank band.
  const sicMat = new THREE.MeshStandardMaterial({roughness: 0.55, map: stripeTex(c => {
    c.fillStyle = "#f2f2ee"; c.fillRect(0, 0, 256, 256);
    c.fillStyle = "#1a1a1a";
    for (let k = 0; k < 4; k += 2) c.fillRect(k * 64, 170, 64, 86);
    c.fillRect(0, 96, 256, 18);
    for (let k = 1; k < 4; k += 2) c.fillRect(k * 64, 0, 64, 30);
  })});
  const sivbMat = new THREE.MeshStandardMaterial({roughness: 0.55, map: stripeTex(c => {
    c.fillStyle = "#f2f2ee"; c.fillRect(0, 0, 256, 256);
    c.fillStyle = "#1a1a1a"; for (let k = 0; k < 4; k += 2) c.fillRect(k * 64, 150, 64, 106);
  })});

  const cyl = (r0, r1, h, mat, y, seg = 48, open = false) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, h, seg, 1, open), mat);
    m.position.y = y + h / 2; return m;
  };
  const bell = (r0, r1, h, y, x = 0, z = 0) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r0, r1, h, 24, 1, true), engineMat);
    m.position.set(x, y - h / 2, z); return m;
  };
  const plume = (len, r, color, y, x = 0, z = 0) => {
    const m = new THREE.Mesh(new THREE.ConeGeometry(r, len, 24, 1, true),
      new THREE.MeshBasicMaterial({color, transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide}));
    m.position.set(x, y - len / 2, z);
    const core = new THREE.Mesh(new THREE.ConeGeometry(r * 0.35, len * 0.45, 16, 1, true),
      new THREE.MeshBasicMaterial({color: 0xfff2cc, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending, depthWrite: false}));
    core.position.y = len * 0.275; m.add(core);
    return m;
  };

  // Heights (m): S-IC 0-42, interstage 42-47.5, S-II 47.5-67.5, adapter 67.5-70.5, S-IVB 70.5-85.5,
  // IU 85.5-86.4, SLA 86.4-94.9, SM 94.9-98.8, CM 98.8-102.0, LES 102-112.
  const R1 = 5.05, R3 = 3.3, RS = 1.96;
  const sic = new THREE.Group();
  sic.add(cyl(R1, R1, 42, sicMat, 0));
  const outer = [[3.3, 0], [-3.3, 0], [0, 3.3], [0, -3.3]].map(([x, z]) => [x * 0.75 + z * 0.75, z * 0.75 - x * 0.75]);
  const f1 = [[0, 0], ...outer];
  f1.forEach(([x, z]) => sic.add(bell(0.9, 1.88, 5.8, 0, x, z)));
  outer.forEach(([x, z]) => {
    const fair = new THREE.Mesh(new THREE.ConeGeometry(1.4, 9, 16), white);
    const d = Math.hypot(x, z), ux = x / d, uz = z / d;
    fair.position.set(ux * (R1 + 0.3), 3.6, uz * (R1 + 0.3)); sic.add(fair);
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.25, 4.5, 3.2), black);
    fin.position.set(ux * (R1 + 1.9), 1.8, uz * (R1 + 1.9)); fin.rotation.y = -Math.atan2(uz, ux); sic.add(fin);
  });
  const sicPlumes = f1.map(([x, z]) => plume(60, 9, 0xff7a20, -5.8, x, z));
  sicPlumes.forEach(p => sic.add(p));

  const skirt = cyl(R1, R1, 5.5, white, 42, 48, true);
  const sii = new THREE.Group();
  sii.add(cyl(R1, R1, 20, white, 47.5));
  sii.add(cyl(R1, R1, 1.2, black, 66.3));
  sii.add(cyl(R1, R3, 3, white, 67.5, 48, true));
  const j2 = [[0, 0], [2.4, 0], [-2.4, 0], [0, 2.4], [0, -2.4]];
  j2.forEach(([x, z]) => sii.add(bell(0.4, 1.0, 3.2, 47.5, x, z)));
  const siiPlumes = j2.map(([x, z]) => plume(26, 2.2, 0x88aaff, 44.3, x, z));
  siiPlumes.forEach(p => sii.add(p));

  const sivb = new THREE.Group();
  sivb.add(cyl(R3, R3, 15, sivbMat, 70.5));
  sivb.add(cyl(R3, R3, 0.9, grey, 85.5));
  sivb.add(bell(0.4, 1.0, 3.0, 70.5));
  const sivbPlume = plume(24, 2.0, 0x88aaff, 67.5);
  sivb.add(sivbPlume);

  // SLA: four petals, each a quarter of a cone frustum, hinged at the bottom.
  const slaPetals = [];
  for (let k = 0; k < 4; k++) {
    const g = new THREE.CylinderGeometry(RS, R3, 8.5, 12, 1, true, k * Math.PI / 2, Math.PI / 2);
    g.translate(0, 4.25, 0);
    const petal = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: 0xe8e8e4, roughness: 0.5, side: THREE.DoubleSide}));
    const hinge = new THREE.Group(); hinge.position.y = 86.4; hinge.add(petal);
    hinge.userData.k = k; slaPetals.push(hinge); sivb.add(hinge);
  }

  // LM, stowed upside-up in the SLA: gold descent stage, grey ascent stage, docking port on top.
  const lm = new THREE.Group();
  {
    const ds = new THREE.Mesh(new THREE.CylinderGeometry(2.0, 2.0, 1.7, 8), gold); ds.position.y = 1.2; lm.add(ds);
    const as = new THREE.Mesh(new THREE.BoxGeometry(2.6, 2.2, 2.4), grey); as.position.y = 3.2; lm.add(as);
    const port = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.6, 0.5, 16), silver); port.position.y = 4.55; lm.add(port);
    for (let k = 0; k < 4; k++) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.6, 6), silver);
      const a = Math.PI / 4 + k * Math.PI / 2;
      leg.position.set(1.9 * Math.cos(a), 1.1, 1.9 * Math.sin(a)); leg.rotation.z = 0.25 * Math.cos(a); leg.rotation.x = -0.25 * Math.sin(a);
      lm.add(leg);
    }
  }
  lm.position.y = 87.0;
  sivb.add(lm);

  // CSM: SPS bell, SM, CM, docking probe. Its own origin at the SM base so it can turn round.
  const csm = new THREE.Group();
  csm.add(bell(0.5, 1.25, 2.6, 0));
  csm.add(cyl(RS, RS, 3.9, silver, 0));
  csm.add(cyl(RS, 0.45, 3.2, silver, 3.9));
  const probe = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.15, 0.9, 8), grey); probe.position.y = 7.55; csm.add(probe);
  const quads = [0, 1, 2, 3].map(k => {
    const q = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.9, 0.5), grey);
    const a = k * Math.PI / 2; q.position.set((RS + 0.2) * Math.cos(a), 2.6, (RS + 0.2) * Math.sin(a)); return q;
  });
  quads.forEach(q => csm.add(q));
  const spsPlume = plume(14, 1.6, 0xccddff, -2.6);
  csm.add(spsPlume);
  const CSM_C = 2.7;                      // pivot: middle of the CSM, 2.7 m above the SM base
  csm.position.y = -CSM_C;
  const csmPivot = new THREE.Group();
  csmPivot.add(csm);
  csmPivot.position.y = 94.9 + CSM_C;

  const les = new THREE.Group();
  les.add(cyl(0.9, 0.35, 4.4, new THREE.MeshStandardMaterial({color: 0xb04a2a, roughness: 0.7, wireframe: true}), 102));
  les.add(cyl(0.33, 0.33, 4.8, white, 106.4));
  les.add(cyl(0.33, 0.05, 0.8, black, 111.2));

  const stack = new THREE.Group();
  [sic, skirt, sii, sivb, csmPivot, les].forEach(o => stack.add(o));
  scene.add(stack);

  // ---- Trajectory helpers ----
  const tr = data.track, times = tr.map(d => d[0]);
  const lerpRow = t => {
    let i = Math.max(0, Math.min(tr.length - 2, d3.bisectRight(times, t) - 1));
    const a = tr[i], b = tr[i + 1], f = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0])));
    return [0, 1, 2].map(j => (a[j + 1] + f * (b[j + 1] - a[j + 1])) * 1000);
  };
  const tabAt = (tab, t) => {
    let i = Math.max(0, Math.min(tab.length - 2, d3.bisectRight(tab.map(d => d[0]), t) - 1));
    const a = tab[i], b = tab[i + 1], f = (t - a[0]) / (b[0] - a[0]);
    return [1, 2, 3].map(j => a[j] + f * (b[j] - a[j]));
  };
  const gmst0 = data.meta.gmst0_rad;
  const T0R = Math.hypot(tr[0][1], tr[0][2], tr[0][3]) * 1000;   // vehicle reference radius on the pad
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  // Direction of travel used for attitude: Earth-relative velocity in the atmosphere,
  // inertial velocity once in orbit, local vertical on the pad.
  const attitude = t => {
    const p0 = V(...lerpRow(t - 1)), p1 = V(...lerpRow(t + 1)), p = V(...lerpRow(t));
    const vin = p1.clone().sub(p0).multiplyScalar(0.5);
    const up = p.clone().normalize();
    const vrel = vin.clone().sub(V(-OMEGA * p.y, OMEGA * p.x, 0));
    let fwd;
    if (t < 1) fwd = up.clone();
    else {
      const w = Math.max(0, Math.min(1, (t - 600) / 120));        // relative -> inertial by orbit insertion
      fwd = vrel.clone().normalize().multiplyScalar(1 - w).add(vin.clone().normalize().multiplyScalar(w)).normalize();
      if (t < 12) fwd.lerp(up, 1 - t / 12).normalize();            // still clearing the tower
    }
    return {p, up, fwd};
  };

  // ---- Camera control: preset + drag to orbit + wheel to zoom ----
  let preset = "Chase", az = 0, el = 0, zoom = 1;
  const presets = {
    Chase: {d: 1, a: -2.1, e: 0.3},
    Side: {d: 1, a: -1.57, e: 0.05},
    Front: {d: 1, a: 0.5, e: 0.25},
    Wide: {d: 60, a: -2.6, e: 0.5},
  };
  const el$ = renderer.domElement;
  el$.style.cssText = "display:block;width:100%;height:auto;border-radius:6px;touch-action:none;cursor:grab";
  let drag = null;
  el$.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; el$.setPointerCapture(e.pointerId); });
  el$.addEventListener("pointermove", e => {
    if (!drag) return;
    az += (e.clientX - drag[0]) * 0.008; el = Math.max(-1.4, Math.min(1.4, el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; dirty = true;
  });
  el$.addEventListener("pointerup", () => { drag = null; });
  el$.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.15, Math.min(400, zoom * Math.exp(e.deltaY * 0.001))); dirty = true; }, {passive: false});

  // ---- Per-time update ----
  let T = -20, dirty = true, label = "";

  function update(t) {
    T = t;
    const {p, up, fwd} = attitude(t);
    // Orientation: +y along fwd, +z towards local up projected perpendicular to fwd.
    let zAxis = up.clone().sub(fwd.clone().multiplyScalar(up.dot(fwd)));
    if (zAxis.lengthSq() < 1e-6) zAxis = V(0, 0, 1).sub(fwd.clone().multiplyScalar(fwd.z));
    zAxis.normalize();
    const xAxis = new THREE.Vector3().crossVectors(fwd, zAxis).normalize();
    const basis = new THREE.Matrix4().makeBasis(xAxis, fwd, zAxis);
    stack.quaternion.setFromRotationMatrix(basis);

    // Stage visibility and separations (offsets along the stack axis, metres).
    const fl = 0.9 + 0.1 * Math.sin(t * 47) * Math.sin(t * 13);
    sicPlumes.forEach(pl => { pl.visible = t > -8.9 && t < 161.6; pl.scale.set(fl, t < 0 ? 0.5 + 0.5 * (t + 8.9) / 8.9 : fl, fl); });
    siiPlumes.forEach(pl => { pl.visible = t > 163.5 && t < 548.2; pl.scale.set(fl, fl, fl); });
    sivbPlume.visible = (t > 552 && t < 699.3) || (t > EVENTS.tliIgn && t < EVENTS.tliCut);
    spsPlume.visible = t > EVENTS.evasive && t < EVENTS.evasive + 3;

    const sep = (obj, t0, dir, a, v0, spin, axis, life = 90) => {
      if (t < t0) { obj.visible = true; obj.position.y = obj.userData.y0 ?? (obj.userData.y0 = obj.position.y); obj.rotation.set(0, 0, 0); return; }
      obj.userData.y0 ??= obj.position.y;
      const dt = t - t0; obj.visible = dt < life;
      obj.position.y = obj.userData.y0 + dir * (v0 * dt + 0.5 * a * dt * dt);
      obj.rotation[axis] = spin * dt;
    };
    sep(sic, EVENTS.sicSep, -1, 9, 2, 0.02, "x", 60);
    sep(skirt, EVENTS.skirtSep, -1, 9.5, 1, 0.08, "z", 40);
    sep(les, EVENTS.tower, +1, -9, 60, 0.15, "x", 20);
    sep(sii, EVENTS.siiSep, -1, 7, 1, 0.03, "x", 60);

    // Transposition, docking and extraction, relative to the S-IVB.
    const s0 = EVENTS.csmSep, turnA = s0 + 30, turnB = s0 + 330, approach = EVENTS.dock;
    slaPetals.forEach(h => {
      const k = h.userData.k, a = k * Math.PI / 2 + Math.PI / 4;
      if (t < s0) { h.visible = true; h.rotation.set(0, 0, 0); h.position.set(0, 86.4, 0); return; }
      const dt = t - s0, open = Math.min(1, dt / 4) * 0.8;
      h.rotation.set(Math.sin(a) * open, 0, -Math.cos(a) * open);
      h.position.set(Math.cos(a) * dt * 0.5, 86.4, Math.sin(a) * dt * 0.5);
      h.visible = dt < 120;
    });
    // CSM pivot height (m): stowed 97.6; out 30 m; turn 180 deg; back in until the probe tip
    // (5.3 m from the pivot once turned) meets the LM docking port at 91.8 m.
    const stowed = 94.9 + CSM_C, docked = 91.8 + (8.0 - CSM_C), gap = 30;
    let py = stowed, flip = 0;
    if (t >= s0 && t < turnA) py = stowed + gap * (t - s0) / (turnA - s0);
    else if (t >= turnA && t < turnB) { py = stowed + gap; flip = (t - turnA) / (turnB - turnA); }
    else if (t >= turnB && t < approach) { flip = 1; py = stowed + gap + (docked - stowed - gap) * (t - turnB) / (approach - turnB); }
    else if (t >= approach) { flip = 1; py = docked; }
    let away = 0;
    if (t >= EVENTS.ejection) {
      away = 0.25 * (t - EVENTS.ejection) + (t > EVENTS.evasive ? 6.0 * Math.max(0, t - EVENTS.evasive - 1.5) : 0);
    }
    csmPivot.position.y = py + (t >= approach ? away : 0);
    csmPivot.rotation.set(0, 0, Math.PI * Math.min(1, flip));
    lm.position.y = 87.0 + away;
    const camTarget = t < EVENTS.sicSep ? V(0, 55, 0) : t < EVENTS.siiSep ? V(0, 80, 0)
      : t < s0 ? V(0, 88, 0) : V(0, csmPivot.position.y - 3, 0);

    // Floating origin: everything relative to the camera target's world position.
    const targetWorld = camTarget.clone().applyQuaternion(stack.quaternion);  // offset from the trajectory point
    // The trajectory reference point sits ~59 m above the pad; raise the Earth to meet the S-IC base.
    const lift = up.clone().multiplyScalar(T0R - padR);
    stack.position.copy(targetWorld.clone().negate());
    const pv = p.clone();                         // vehicle reference point (m)
    earth.position.copy(pv.clone().add(targetWorld).negate().add(lift));
    earth.rotation.set(0, 0, gmst0 + OMEGA * t);
    const m = V(...tabAt(data.moon, t)).multiplyScalar(1000);
    moon.position.copy(m.sub(pv).sub(targetWorld));
    const s = V(...tabAt(data.sun, t)).normalize();
    sun.position.copy(s.clone().multiplyScalar(1e7)); sun.target.position.set(0, 0, 0);
    // Sky: blue in daylight near the ground, fading to black by about 60 km.
    const alt = p.length() - padR, sunEl = s.dot(up);
    const day = Math.max(0, Math.min(1, (sunEl + 0.1) / 0.3)) * Math.max(0, 1 - alt / 60000);
    scene.background.setRGB(0.01 + 0.40 * day, 0.016 + 0.62 * day, 0.04 + 0.86 * day);
    stars.visible = day < 0.25;

    // Camera
    const pr = presets[preset];
    const baseD = t < EVENTS.sicSep ? 260 : t < EVENTS.siiSep ? 160 : t < s0 ? 110 : 70;
    const d = baseD * pr.d * zoom;
    const a = pr.a + az, e = pr.e + el;
    // camera offset in the stack frame: x side, y along axis, z "up"
    const offLocal = V(Math.cos(e) * Math.sin(a), Math.cos(e) * Math.cos(a), Math.sin(e)).multiplyScalar(d);
    const off = offLocal.applyQuaternion(stack.quaternion);
    camera.position.copy(off);
    // Keep the camera above the ground (matters on the pad and in the first seconds of flight).
    const fromC = camera.position.clone().sub(earth.position);
    const sl = fromC.z / fromC.length();
    const ground = RE * (1 - FLAT * sl * sl) + 25;
    if (fromC.length() < ground) camera.position.copy(earth.position).add(fromC.setLength(ground));
    camera.up.copy(up);                               // keep the horizon level
    camera.lookAt(0, 0, 0);
    dirty = true;
  }

  let raf = null;
  const loop = () => {
    if (dirty || (T > -9 && T < 10300)) { if (!dirty) update(T); renderer.render(scene, camera); dirty = false; }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);

  const setPreset = name => { preset = name; az = 0; el = 0; zoom = 1; update(T); };
  return {
    canvas: el$, update, setPreset, presets: Object.keys(presets),
    dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); },
  };
}
