// CGI views for the Part 4 player: lunar ascent, rendezvous and docking, transearth injection,
// the coast home and entry. Spacecraft are built from primitives at real dimensions (metres).
// World frame: geocentric, mean equator and equinox of 16 July 1969, with a floating origin at the
// spacecraft the camera is following. The Moon and Earth are textured spheres turned by their
// orientation tables; near Tranquility Base a local terrain mesh with illustrative craters is added.

export const MARIA = [
  ["Oceanus Procellarum", 20, -57, 30], ["Oceanus Procellarum", -2, -45, 14], ["Mare Imbrium", 33, -16, 18],
  ["Mare Frigoris", 56, -10, 6], ["Mare Frigoris", 57, 10, 6], ["Mare Frigoris", 55, 28, 5],
  ["Mare Serenitatis", 28, 17.5, 10.5], ["Mare Tranquillitatis", 8.5, 31, 12], ["Mare Tranquillitatis", 13, 25, 6],
  ["Mare Tranquillitatis", 3, 24, 4], ["Mare Crisium", 17, 59, 8], ["Mare Fecunditatis", -7.8, 51.3, 10],
  ["Mare Nectaris", -15.2, 35.5, 5.5], ["Mare Nubium", -21, -17, 11], ["Mare Cognitum", -10, -23, 6],
  ["Mare Humorum", -24.4, -38.6, 6.5], ["Mare Vaporum", 13.3, 3.6, 4], ["Sinus Medii", 2, 1, 3.5],
  ["Mare Insularum", 7.5, -30.9, 7], ["Mare Smythii", 1.3, 87.5, 6], ["Mare Marginis", 13.3, 86.1, 5],
  ["Mare Moscoviense", 27.3, 147.9, 4.5], ["Mare Orientale", -19.4, -92.8, 5], ["Mare Australe", -38.9, 93, 8],
  ["Mare Ingenii", -33.7, 163.5, 4],
];
export const CRATERS = [
  ["Maskelyne", 2.2, 30.1, 23], ["Moltke", -0.6, 24.2, 6.5], ["Sabine", 1.4, 20.1, 30], ["Ritter", 2.0, 19.2, 29],
  ["Schmidt", 0.9, 18.8, 11], ["Arago", 6.2, 21.4, 26], ["Censorinus", -0.4, 32.7, 3.8], ["Secchi", 2.4, 43.5, 22],
  ["Taruntius", 5.5, 46.5, 56], ["Torricelli", -4.6, 28.5, 22], ["Theon Junior", -2.4, 15.8, 18],
];

export function createScene(THREE, {data, land, d3, width, height}) {
  const W = width, H = height, M = data.meta, RAD = Math.PI / 180;
  const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H);
  renderer.outputEncoding = THREE.sRGBEncoding;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 5e10);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  scene.add(sun, sun.target);
  const ambient = new THREE.AmbientLight(0x8899aa, 0.35);
  scene.add(ambient);
  {
    const g = new THREE.BufferGeometry(), n = 3000, p = new Float32Array(n * 3);
    let seed = 13; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) {
      const z = 2 * rnd() - 1, a = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
      p.set([2e10 * r * Math.cos(a), 2e10 * r * Math.sin(a), 2e10 * z], 3 * i);
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    var stars = new THREE.Points(g, new THREE.PointsMaterial({color: 0xffffff, size: 1.1, sizeAttenuation: false}));
    scene.add(stars);
  }

  // ---- lookups -------------------------------------------------------------------------
  const tabAt = (rows, t, n) => {
    const ts = rows._t ?? (rows._t = rows.map(d => d[0]));
    const i = Math.max(0, Math.min(rows.length - 2, d3.bisectRight(ts, t) - 1));
    const a = rows[i], b = rows[i + 1], f = Math.max(0, Math.min(1, (t - a[0]) / ((b[0] - a[0]) || 1)));
    return d3.range(1, n + 1).map(j => a[j] + f * (b[j] - a[j]));
  };
  const moonPos = t => V(...tabAt(data.moon, t, 3)).multiplyScalar(1000);         // geocentric, m
  const lmPos = t => V(...tabAt(data.lm, t, 3)).multiplyScalar(1000).add(moonPos(t));
  const csmLunar = t => V(...tabAt(data.csm, t, 3)).multiplyScalar(1000).add(moonPos(t));
  const earthRows = data.earth, entryRows = data.entry;
  const tEarth0 = earthRows[0][0], tEntry0 = entryRows[0][0];
  const csmPos = t => {
    if (t < M.tei[1] + 600) return csmLunar(t);
    if (t < tEntry0) return V(...tabAt(earthRows, t, 3)).multiplyScalar(1000);
    return V(...tabAt(entryRows, t, 3)).multiplyScalar(1000);
  };
  const vel = (f, t, h = 2) => f(t + h).sub(f(t - h)).multiplyScalar(1 / (2 * h));
  const bodyMatrix = t => {
    const o = tabAt(data.moon_orient, t, 9);
    const bx = V(o[0], o[3], o[6]).normalize(), bz = V(o[2], o[5], o[8]);
    bz.sub(bx.clone().multiplyScalar(bz.dot(bx))).normalize();
    return new THREE.Matrix4().makeBasis(bx, new THREE.Vector3().crossVectors(bz, bx), bz);
  };
  const gmst = t => M.gmst0 + M.omega_earth * t;
  const sunDir = t => { const s = tabAt(data.sun, t, 8); return V(s[5], s[6], s[7]).normalize(); };

  // ---- textured spheres in body frames --------------------------------------------------
  const latLonSphere = (R, map, nLat = 90, nLon = 180) => {
    const pos = [], uv = [], idx = [];
    for (let i = 0; i <= nLat; i++) {
      const lat = -90 + 180 * i / nLat;
      for (let j = 0; j <= nLon; j++) {
        const lon = -180 + 360 * j / nLon;
        pos.push(R * Math.cos(lat * RAD) * Math.cos(lon * RAD), R * Math.cos(lat * RAD) * Math.sin(lon * RAD), R * Math.sin(lat * RAD));
        uv.push(j / nLon, i / nLat);
      }
    }
    for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
      const a = i * (nLon + 1) + j, b = a + 1, c = a + nLon + 1, d = c + 1;
      idx.push(a, b, c, b, d, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    g.setIndex(idx); g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({map, roughness: 1, metalness: 0}));
    m.matrixAutoUpdate = false;
    return m;
  };
  const canvasTex = (w, h, draw) => {
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
    draw(cv.getContext("2d"), w, h);
    const tex = new THREE.CanvasTexture(cv); tex.encoding = THREE.sRGBEncoding; tex.anisotropy = 4;
    return tex;
  };
  const proj = (w, h) => d3.geoEquirectangular().scale(w / (2 * Math.PI)).translate([w / 2, h / 2]).reflectY(true);
  const moonTex = canvasTex(2048, 1024, (c, w, h) => {
    c.fillStyle = "#9c9b97"; c.fillRect(0, 0, w, h);
    const path = d3.geoPath(proj(w, h), c);
    c.fillStyle = "#6e6f73";
    for (const [, lat, lon, r] of MARIA) { c.beginPath(); path(d3.geoCircle().center([lon, lat]).radius(r)()); c.fill(); }
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let k = 0; k < 1400; k++) {
      const lon = 360 * rnd() - 180, lat = Math.asin(2 * rnd() - 1) / RAD, r = 0.15 + 2.2 * Math.pow(rnd(), 3);
      c.beginPath(); path(d3.geoCircle().center([lon, lat]).radius(r)());
      c.strokeStyle = "rgba(255,255,255,0.10)"; c.lineWidth = 1; c.stroke();
      c.fillStyle = "rgba(0,0,0,0.06)"; c.fill();
    }
  });
  const moon = latLonSphere(1737.4e3, moonTex);
  scene.add(moon);
  const earthTex = canvasTex(2048, 1024, (c, w, h) => {
    c.fillStyle = "#1d4e89"; c.fillRect(0, 0, w, h);
    if (land) { const path = d3.geoPath(proj(w, h), c); c.beginPath(); path(land); c.fillStyle = "#5b7a4a"; c.fill(); }
  });
  const earth = latLonSphere(6378.137e3, earthTex, 120, 240);
  scene.add(earth);
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(6378.137e3 + 60e3, 96, 48),
    new THREE.MeshBasicMaterial({color: 0x6fa8ff, transparent: true, opacity: 0.12, side: THREE.BackSide, depthWrite: false}));
  scene.add(atmo);

  // ---- local terrain at Tranquility Base (body frame, relative to the site point) --------
  const RS = M.r_site * 1000, site = M.landing_site;
  const unit = (lat, lon) => V(Math.cos(lat * RAD) * Math.cos(lon * RAD), Math.cos(lat * RAD) * Math.sin(lon * RAD), Math.sin(lat * RAD));
  const S = unit(site[0], site[1]);
  const E = V(-Math.sin(site[1] * RAD), Math.cos(site[1] * RAD), 0);
  const N = new THREE.Vector3().crossVectors(S, E);
  const craters = [];
  {
    let seed = 29; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    craters.push([-400 * 0.995, -400 * 0.1, 95, 28]);        // West crater, roughly uprange (east)
    for (let i = 0; i < 400; i++) {
      const r = 40 + 4000 * Math.sqrt(rnd()), a = 2 * Math.PI * rnd(), R = 2 + 45 * Math.pow(rnd(), 3);
      craters.push([r * Math.cos(a), r * Math.sin(a), R, 0.22 * R]);
    }
    for (let i = 0; i < 220; i++) {
      const r = 3000 + 600000 * Math.pow(rnd(), 1.6), a = 2 * Math.PI * rnd(), R = 150 + 3500 * Math.pow(rnd(), 3);
      craters.push([r * Math.cos(a), r * Math.sin(a), R, 0.18 * R]);
    }
    for (const [, lat, lon, dkm] of CRATERS) {
      const u = unit(lat, lon), ang = Math.acos(Math.min(1, u.dot(S)));
      const tt = u.clone().sub(S.clone().multiplyScalar(u.dot(S))).normalize(), d = ang * RS;
      craters.push([d * tt.dot(E), d * tt.dot(N), dkm * 500, Math.min(2500, dkm * 500 * 0.18)]);
    }
  }
  craters[0][0] = 400; craters[0][1] = 0;     // West crater lies east of the site (Eagle came in from the east)
  const heightAt = (x, y) => {
    let h = 0;
    for (const [cx, cy, R, D] of craters) {
      const dx = x - cx, dy = y - cy;
      if (Math.abs(dx) > 1.6 * R || Math.abs(dy) > 1.6 * R) continue;
      const r = Math.hypot(dx, dy) / R;
      if (r < 1) h += -D * (1 - r * r) + 0.08 * D;
      else if (r < 1.6) h += 0.08 * D * Math.exp(-(((r - 1) / 0.25) ** 2));
    }
    return h;
  };
  const surface = new THREE.Group();
  {
    const nR = 160, nA = 240, r0 = 0.8, r1 = 600000, q = Math.pow(r1 / r0, 1 / (nR - 1));
    const pos = new Float32Array((nR * nA + 1) * 3), col = new Float32Array((nR * nA + 1) * 3), idx = [];
    const put = (k, x, y) => {
      const rho = Math.hypot(x, y), ang = rho / RS;
      const t = rho > 0 ? E.clone().multiplyScalar(x / rho).add(N.clone().multiplyScalar(y / rho)) : E.clone();
      const u = S.clone().multiplyScalar(Math.cos(ang)).add(t.multiplyScalar(Math.sin(ang)));
      const p = u.clone().multiplyScalar(RS + heightAt(x, y)).sub(S.clone().multiplyScalar(RS));
      pos.set([p.x, p.y, p.z], 3 * k);
      const b = 0.42 * (0.92 + 0.08 * Math.sin(x * 0.0007 + 1.3 * Math.cos(y * 0.0009)));
      col.set([b, b * 0.985, b * 0.95], 3 * k);
    };
    put(0, 0, 0);
    for (let i = 0; i < nR; i++) for (let j = 0; j < nA; j++) {
      const r = r0 * Math.pow(q, i), a = 2 * Math.PI * j / nA;
      put(1 + i * nA + j, r * Math.cos(a), r * Math.sin(a));
    }
    for (let j = 0; j < nA; j++) idx.push(0, 1 + j, 1 + (j + 1) % nA);
    for (let i = 0; i < nR - 1; i++) for (let j = 0; j < nA; j++) {
      const a = 1 + i * nA + j, b = 1 + i * nA + (j + 1) % nA, c = a + nA, d = b + nA;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx); g.computeVertexNormals();
    surface.add(new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors: true, roughness: 1})));
  }
  surface.matrixAutoUpdate = false;
  scene.add(surface);

  // ---- spacecraft ------------------------------------------------------------------------
  const gold = new THREE.MeshStandardMaterial({color: 0xc9a227, roughness: 0.35, metalness: 0.7});
  const black = new THREE.MeshStandardMaterial({color: 0x222222, roughness: 0.7});
  const grey = new THREE.MeshStandardMaterial({color: 0xa9adb3, roughness: 0.5, metalness: 0.3});
  const silver = new THREE.MeshStandardMaterial({color: 0xd8dade, roughness: 0.25, metalness: 0.85});
  const dark = new THREE.MeshStandardMaterial({color: 0x3a3d42, roughness: 0.6, metalness: 0.2});
  const shield = new THREE.MeshStandardMaterial({color: 0x5a4632, roughness: 0.9});
  const addMesh = (grp, geo, mat, x = 0, y = 0, z = 0) => { const m = new THREE.Mesh(geo, mat); m.position.set(x, y, z); grp.add(m); return m; };
  // Descent stage, left on the surface. Local axes: +y up.
  const descent = new THREE.Group();
  {
    const LEG = 2.6;
    const ds = addMesh(descent, new THREE.CylinderGeometry(2.15, 2.15, 1.65, 8), gold, 0, LEG + 0.83);
    ds.rotation.y = Math.PI / 8;
    for (let k = 0; k < 4; k++) {
      const a = Math.PI / 4 + k * Math.PI / 2, top = V(1.9 * Math.cos(a), LEG + 1.2, 1.9 * Math.sin(a)), pad = V(4.3 * Math.cos(a), 0.1, 4.3 * Math.sin(a));
      const st = top.clone().sub(pad), s = addMesh(descent, new THREE.CylinderGeometry(0.07, 0.07, st.length(), 8), gold);
      s.position.copy(top.clone().add(pad).multiplyScalar(0.5)); s.quaternion.setFromUnitVectors(V(0, 1, 0), st.normalize());
      addMesh(descent, new THREE.CylinderGeometry(0.47, 0.47, 0.18, 16), silver, pad.x, pad.y, pad.z);
    }
    // the flag, about 8 m from the LM
    addMesh(descent, new THREE.CylinderGeometry(0.02, 0.02, 2.3, 6), silver, -8, 1.15, 2);
    addMesh(descent, new THREE.BoxGeometry(0.9, 0.6, 0.02), new THREE.MeshStandardMaterial({color: 0xdddddd}), -7.55, 2.0, 2);
  }
  descent.matrixAutoUpdate = false;
  scene.add(descent);
  // Ascent stage: local +y = LM +X (thrust axis, up), -z = LM +Z (forward, windows), +x = LM +Y.
  const ascent = new THREE.Group();
  {
    const y0 = 0;
    addMesh(ascent, new THREE.BoxGeometry(3.0, 2.3, 2.6), grey, 0, y0 + 1.15, -0.1);
    addMesh(ascent, new THREE.BoxGeometry(2.2, 1.0, 1.2), dark, 0, y0 + 2.6, 0.7);
    addMesh(ascent, new THREE.CylinderGeometry(0.45, 0.5, 0.35, 16), silver, 0, y0 + 2.45, 0);   // docking tunnel and drogue
    [-0.55, 0.55].forEach(x => { const w = addMesh(ascent, new THREE.CircleGeometry(0.32, 3), black, x, y0 + 1.75, -1.42); w.rotation.y = Math.PI; });
    addMesh(ascent, new THREE.BoxGeometry(0.85, 0.85, 0.05), dark, 0, y0 + 0.6, -1.42);
    [[1.6, 1.4], [-1.6, 1.4], [1.6, -1.6], [-1.6, -1.6]].forEach(([x, z]) => addMesh(ascent, new THREE.BoxGeometry(0.35, 0.35, 0.35), grey, x, y0 + 1.9, z));
    addMesh(ascent, new THREE.CylinderGeometry(0.03, 0.03, 1.2, 6), silver, 1.2, y0 + 2.9, 0.9);      // rendezvous radar mast
    addMesh(ascent, new THREE.SphereGeometry(0.35, 12, 8), silver, 1.2, y0 + 3.5, 0.9);
    const bell = addMesh(ascent, new THREE.CylinderGeometry(0.3, 0.45, 0.9, 16, 1, true),
      new THREE.MeshStandardMaterial({color: 0x2b2b2b, side: THREE.DoubleSide}), 0, y0 - 0.3, 0);
  }
  const plumeMat = () => new THREE.MeshBasicMaterial({color: 0xd8e4ff, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide});
  const apsPlume = addMesh(ascent, new THREE.ConeGeometry(1.1, 7, 20, 1, true), plumeMat(), 0, -0.8 - 3.5, 0);
  ascent.matrixAutoUpdate = true;
  scene.add(ascent);
  // Columbia: local +y = CSM +X (nose, docking probe). SM below, SPS bell at the bottom.
  const csm = new THREE.Group();
  const cm = new THREE.Group();
  {
    const cyl = (grp, r0, r1, h, mat, y) => addMesh(grp, new THREE.CylinderGeometry(r1, r0, h, 40), mat, 0, y + h / 2);
    cyl(csm, 1.96, 1.96, 3.9, silver, -3.9);                                 // service module
    const bell = addMesh(csm, new THREE.CylinderGeometry(0.5, 1.25, 2.9, 20, 1, true),
      new THREE.MeshStandardMaterial({color: 0x2b2b2b, side: THREE.DoubleSide}), 0, -3.9 - 1.45, 0);
    addMesh(csm, new THREE.BoxGeometry(0.4, 1.4, 0.4), silver, 1.6, -2.5, 1.6);   // high-gain boom (stylised)
    cyl(cm, 1.96, 0.45, 3.2, silver, 0);                                      // command module cone
    addMesh(cm, new THREE.CylinderGeometry(1.96, 1.96, 0.25, 40), shield, 0, -0.12, 0);   // heat shield
    addMesh(cm, new THREE.CylinderGeometry(0.06, 0.06, 0.6, 8), silver, 0, 3.45, 0);     // docking probe
    csm.add(cm);
  }
  const spsPlume = addMesh(csm, new THREE.ConeGeometry(1.6, 12, 24, 1, true), plumeMat(), 0, -3.9 - 2.9 - 6, 0);
  spsPlume.rotation.x = Math.PI;
  scene.add(csm);
  // Entry: plasma sheath and parachutes, attached to the CM.
  const glowTex = canvasTex(256, 256, (c, w, h) => {
    const gr = c.createRadialGradient(128, 128, 5, 128, 128, 128);
    gr.addColorStop(0, "rgba(255,240,220,1)"); gr.addColorStop(0.3, "rgba(255,170,90,0.75)");
    gr.addColorStop(0.65, "rgba(255,90,40,0.25)"); gr.addColorStop(1, "rgba(255,60,20,0)");
    c.fillStyle = gr; c.fillRect(0, 0, w, h);
  });
  const plasma = new THREE.Sprite(new THREE.SpriteMaterial({map: glowTex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true}));
  plasma.scale.set(14, 14, 1);
  scene.add(plasma);
  const trail = new THREE.Mesh(new THREE.ConeGeometry(2.4, 40, 24, 1, true),
    new THREE.MeshBasicMaterial({color: 0xff8a40, transparent: true, opacity: 0.25, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide}));
  scene.add(trail);
  const chuteMat = (c) => new THREE.MeshStandardMaterial({color: c, roughness: 0.8, side: THREE.DoubleSide});
  const chutes = new THREE.Group();
  const drogues = [], mains = [];
  for (let k = 0; k < 2; k++) {
    const g = new THREE.Group();
    const canopy = addMesh(g, new THREE.SphereGeometry(2.5, 20, 8, 0, Math.PI * 2, 0, Math.PI / 2.3), chuteMat(0xffffff));
    canopy.position.y = 30;
    g.rotation.z = (k ? 1 : -1) * 0.18;
    chutes.add(g); drogues.push(g);
  }
  for (let k = 0; k < 3; k++) {
    const g = new THREE.Group();
    const canopy = addMesh(g, new THREE.SphereGeometry(12.7, 28, 10, 0, Math.PI * 2, 0, Math.PI / 2.4), chuteMat(k === 1 ? 0xffffff : 0xe8673c));
    canopy.position.y = 45;
    for (let s = 0; s < 8; s++) {
      const a = s * Math.PI / 4, top = V(11 * Math.cos(a), 43, 11 * Math.sin(a)), st = top.clone();
      const line = addMesh(g, new THREE.CylinderGeometry(0.03, 0.03, st.length(), 4), chuteMat(0xcccccc));
      line.position.copy(top.clone().multiplyScalar(0.5)); line.quaternion.setFromUnitVectors(V(0, 1, 0), st.normalize());
    }
    g.rotation.z = (k - 1) * 0.32; g.rotation.x = k === 1 ? 0.3 : -0.12;
    chutes.add(g); mains.push(g);
  }
  scene.add(chutes);
  // Ocean patch near the splash point (Earth-fixed, rotates with Earth).
  const ocean = new THREE.Mesh(new THREE.CircleGeometry(60e3, 64), new THREE.MeshStandardMaterial({color: 0x123c66, roughness: 0.35, metalness: 0.1}));
  scene.add(ocean);

  // Trajectory lines (positions relative to the floating origin are rebuilt each update).
  const mkLine = (n, color, opacity = 0.8) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({color, transparent: true, opacity}));
    l.frustumCulled = false; scene.add(l); return l;
  };
  const lmLine = mkLine(1500, 0xf4a261), csmLine = mkLine(1500, 0x6fa8dc), homeLine = mkLine(1500, 0x9ad1ff, 0.7), moonLine = mkLine(400, 0x888888, 0.5);
  const setLine = (line, pts, origin) => {
    const a = line.geometry.attributes.position, n = Math.min(pts.length, a.count);
    for (let i = 0; i < a.count; i++) {
      const p = pts[Math.min(i, n - 1)] ?? origin;
      a.setXYZ(i, p.x - origin.x, p.y - origin.y, p.z - origin.z);
    }
    a.needsUpdate = true;
    line.geometry.computeBoundingSphere();
  };

  // ---- attitude helpers ------------------------------------------------------------------
  const basis = (Y, Zf) => {          // local +y along Y, local -z along Zf (forward)
    const y = Y.clone().normalize();
    const f = Zf.clone().sub(y.clone().multiplyScalar(Zf.dot(y))).normalize();
    const x = new THREE.Vector3().crossVectors(y, f.clone().negate());
    return new THREE.Matrix4().makeBasis(x, y, f.clone().negate());
  };
  const lmAccel = t => vel(lmPos, t + 1, 1).sub(vel(lmPos, t - 1, 1)).multiplyScalar(0.5);
  const gravMoon = (p, t) => { const r = p.clone().sub(moonPos(t)); return r.multiplyScalar(-4.9028e12 / Math.pow(r.length(), 3)); };
  const lmAttitude = t => {
    const p = lmPos(t), up = p.clone().sub(moonPos(t)).normalize();
    if (t < M.liftoff + 0.5) {
      const f = vel(csmLunar, t, 5);
      return basis(up, f.lengthSq() ? f : V(1, 0, 0));
    }
    if (t < M.insertion) {
      const thr = lmAccel(t).sub(gravMoon(p, t));
      const v = vel(lmPos, t, 1).sub(vel(moonPos, t, 30));
      const fwdH = v.clone().sub(up.clone().multiplyScalar(v.dot(up)));
      // windows forward and up: the LM flew the ascent face-forward, heads up
      return basis(thr.lengthSq() > 1e-6 ? thr : up, up.clone().multiplyScalar(-1).add(fwdH.normalize().multiplyScalar(0.0)).lengthSq() ? up.clone().negate() : fwdH);
    }
    // After insertion: the LM's +Z (forward) toward Columbia (radar tracking), +X up.
    const toC = csmLunar(t).sub(p);
    return basis(up, toC.lengthSq() > 1 ? toC : vel(lmPos, t, 2));
  };

  // ---- camera ------------------------------------------------------------------------------
  let preset = "Chase", az = 0, el = 0, zoom = 1, T = M.liftoff, dirty = true, info = {label: ""};
  const presets = ["Chase", "Columbia's window", "Wide"];
  const el$ = renderer.domElement;
  el$.style.cssText = "display:block;width:100%;height:auto;border-radius:6px;touch-action:none;cursor:grab";
  let drag = null;
  el$.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; el$.setPointerCapture(e.pointerId); });
  el$.addEventListener("pointermove", e => {
    if (!drag) return;
    az += (e.clientX - drag[0]) * 0.008; el = Math.max(-1.4, Math.min(1.4, el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; update(T);
  });
  el$.addEventListener("pointerup", () => { drag = null; });
  el$.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.05, Math.min(80, zoom * Math.exp(e.deltaY * 0.001))); update(T); }, {passive: false});

  const phaseOf = t => t < M.liftoff ? "surface" : t < M.insertion ? "ascent" : t < M.dock_contact ? "rendezvous"
    : t < M.jettison ? "docked" : t < M.tei[0] - 60 ? "csm" : t < M.tei[1] + 3 * 3600 ? "tei" : t < M.ei - 1500 ? "coast" : "entry";

  function update(t) {
    T = t;
    const ph = phaseOf(t);
    const pm = moonPos(t), B = bodyMatrix(t);
    const pL = t <= M.jettison + 5 ? lmPos(t) : lmPos(Math.min(t, data.lm.at(-1)[0]));
    const pC = csmPos(t);
    const lmFocus = ph === "surface" || ph === "ascent" || ph === "rendezvous";
    const O = lmFocus ? pL.clone() : pC.clone();               // floating origin
    const sd = sunDir(t);
    sun.position.copy(sd.clone().multiplyScalar(1e6)); sun.target.position.set(0, 0, 0);

    // Bodies
    moon.matrix.copy(B).setPosition(pm.clone().sub(O)); moon.matrixWorldNeedsUpdate = true;
    const siteW = S.clone().applyMatrix4(B).multiplyScalar(RS).add(pm);
    const nearMoon = O.distanceTo(pm) < 1737e3 + 400e3;
    surface.visible = nearMoon && O.distanceTo(siteW) < 900e3;
    surface.matrix.copy(B).setPosition(siteW.clone().sub(O)); surface.matrixWorldNeedsUpdate = true;
    const g = gmst(t), Rz = new THREE.Matrix4().makeRotationZ(g);
    earth.matrix.copy(Rz).setPosition(-O.x, -O.y, -O.z); earth.matrixWorldNeedsUpdate = true;
    atmo.position.set(-O.x, -O.y, -O.z);
    // Descent stage at the site, upright.
    const upS = siteW.clone().sub(pm).normalize();
    const eS = E.clone().applyMatrix4(new THREE.Matrix4().extractRotation(B));
    descent.matrix.copy(basis(upS, eS)).setPosition(siteW.clone().sub(O)); descent.matrixWorldNeedsUpdate = true;
    descent.visible = nearMoon && O.distanceTo(siteW) < 300e3;
    // Ocean at the splash point.
    const sl = M.splash_ll, us = unit(sl[0], sl[1]).applyMatrix4(Rz);
    ocean.position.copy(us.clone().multiplyScalar(6378.137e3 * (1 - (1 / 298.257) * Math.sin(sl[0] * RAD) ** 2) + 2).sub(O));
    ocean.quaternion.setFromUnitVectors(V(0, 0, 1), us);
    ocean.visible = ph === "entry" && t > M.drogue - 400;

    // Eagle's ascent stage
    ascent.visible = t <= M.jettison + 1800 && ph !== "coast" && ph !== "entry" && t < M.tei[0];
    if (ascent.visible) {
      const Rl = t > M.dock_contact ? basis(vel(csmLunar, t, 5).normalize().negate(), pL.clone().sub(pm)) : lmAttitude(t);
      ascent.quaternion.setFromRotationMatrix(Rl);
      let pos = pL.clone().sub(O);
      if (t > M.dock_contact && t <= M.jettison) {
        // docked: ascent stage on Columbia's nose
        const Rc = csmAttitude(t);
        const nose = V(0, 1, 0).applyMatrix4(Rc);
        pos = pC.clone().sub(O).add(nose.clone().multiplyScalar(3.2 + 0.4 + 2.8));
        ascent.quaternion.setFromRotationMatrix(basis(nose.clone().negate(), V(1, 0, 0).applyMatrix4(Rc)));
      }
      if (ph === "surface") {
        // sitting on the descent stage
        pos = siteW.clone().sub(O).add(upS.clone().multiplyScalar(2.6 + 1.65));
        ascent.quaternion.setFromRotationMatrix(basis(upS, eS));
      }
      ascent.position.copy(pos);
      apsPlume.visible = ph === "ascent";
      const fl = 0.9 + 0.1 * Math.sin(t * 37) * Math.sin(t * 11);
      apsPlume.scale.set(fl, 1, fl);
    }
    if (ph === "ascent" || ph === "surface") {
      // the ascent stage rises from the descent stage; keep the reconstructed track, offset to the stage top
      if (ph === "ascent") ascent.position.add(upS.clone().multiplyScalar(4.25 * Math.max(0, 1 - (t - M.liftoff) / 3)));
    }

    // Columbia / command module
    const Rc = csmAttitude(t);
    csm.quaternion.setFromRotationMatrix(Rc);
    csm.position.copy(pC.clone().sub(O));
    const sepCM = t >= M.cm_sm_sep;
    csm.children.forEach(ch => { if (ch !== cm) ch.visible = !sepCM; });
    spsPlume.visible = !sepCM && t >= M.tei[0] && t <= M.tei[1];
    // Entry effects
    let gNow = 0, bank = 0, alt = 1e9;
    if (t >= tEntry0) { const r = tabAt(entryRows, t, 12); alt = r[3] * 0.3048; gNow = r[6]; bank = r[7]; }
    const heat = t > M.ei && t < M.drogue ? Math.min(1, gNow / 2.5) * (alt > 30e3 ? 1 : 0) : 0;
    plasma.visible = trail.visible = heat > 0.02;
    if (plasma.visible) {
      const v = vel(csmPos, t, 1).normalize();
      plasma.position.copy(csm.position.clone().add(v.clone().multiplyScalar(2.5)));
      plasma.material.opacity = Math.min(1, heat * 1.2);
      plasma.scale.setScalar(8 + 10 * heat);
      trail.position.copy(csm.position.clone().add(v.clone().multiplyScalar(-18)));
      trail.quaternion.setFromUnitVectors(V(0, 1, 0), v);
      trail.material.opacity = 0.35 * heat;
    }
    chutes.visible = t >= M.drogue && t < M.splash + 600;
    if (chutes.visible) {
      const up = pC.clone().normalize();
      chutes.position.copy(csm.position);
      chutes.quaternion.setFromUnitVectors(V(0, 1, 0), up);
      const fm = Math.max(0, Math.min(1, (t - M.main) / 8));
      drogues.forEach(d => { d.visible = t < M.main + 2; });
      mains.forEach(m => { m.visible = t >= M.main; m.scale.setScalar(0.25 + 0.75 * fm); });
      if (t > M.splash) mains.forEach(m => { m.scale.setScalar(Math.max(0.05, 1 - (t - M.splash) / 30)); });
    }

    // Lines
    const lmPts = [], csmPts = [], home = [];
    if (ph !== "coast" && ph !== "entry" || preset === "Wide") {
      for (let k = 0; k < 200; k++) {
        const tt = t - 7200 + 36 * k;
        if (tt >= M.liftoff && tt <= Math.min(t, M.dock_contact)) lmPts.push(lmPos(tt));
        if (tt <= Math.min(t, M.tei[1] + 600) && tt >= data.csm[0][0]) csmPts.push(csmLunar(tt));
      }
    }
    if (t > M.tei[0]) {
      const a = earthRows, n = a.length, step = Math.max(1, Math.floor(n / 1200));
      for (let i = 0; i < n; i += step) if (a[i][0] <= t) home.push(V(a[i][1], a[i][2], a[i][3]).multiplyScalar(1000));
      if (t > tEntry0) for (const r of entryRows) if (r[0] <= t && r[0] % 10 < 2) home.push(V(r[1], r[2], r[3]).multiplyScalar(1000));
      home.push(pC.clone());
    }
    setLine(lmLine, lmPts.length ? lmPts : [O], O);
    setLine(csmLine, csmPts.length ? csmPts : [O], O);
    setLine(homeLine, home.length ? home : [O], O);
    const mo = [];
    for (let k = 0; k < 360; k++) mo.push(moonPos(t - 3 * 86400 + k * 1800));
    setLine(moonLine, mo, O);
    lmLine.visible = ph !== "entry"; csmLine.visible = ph !== "coast" && ph !== "entry";
    homeLine.visible = t > M.tei[0] && (ph !== "entry" || preset === "Wide");
    moonLine.visible = preset === "Wide" && (ph === "coast" || ph === "tei" || ph === "entry");

    // Labels
    const labels = {surface: "Eagle on the surface", ascent: "Ascent engine firing", rendezvous: "Rendezvous",
      docked: "Docked", csm: "Columbia alone in lunar orbit", tei: t <= M.tei[1] ? "TEI: service propulsion engine firing" : "Leaving the Moon",
      coast: "Coasting home", entry: t < M.ei ? "Approaching entry interface" : t < M.drogue ? "Entry" : t < M.main ? "Drogues" : t < M.splash ? "Main parachutes" : "In the Pacific"};
    info = {label: labels[ph]};

    // Camera
    const focusPos = lmFocus ? ascent.position.clone() : csm.position.clone();
    const upRef = (nearMoon ? O.clone().sub(pm) : O.clone()).normalize();
    let fwd = lmFocus ? vel(lmPos, Math.max(t, M.liftoff + 1), 2).sub(vel(moonPos, t, 30)) : vel(csmPos, t, 2).sub(nearMoon ? vel(moonPos, t, 30) : V(0, 0, 0));
    if (fwd.lengthSq() < 1e-6) fwd = new THREE.Vector3().crossVectors(upRef, V(0, 0, 1));
    fwd.sub(upRef.clone().multiplyScalar(fwd.dot(upRef))).normalize();
    const side = new THREE.Vector3().crossVectors(fwd, upRef).normalize();
    camera.up.copy(upRef);
    if (preset === "Wide") {
      // pull far back: the Moon and the orbits, or the Earth-Moon system
      const big = ph === "coast" || (ph === "tei" && t > M.tei[1] + 1800) ? 6e8 : ph === "entry" && t < M.ei - 600 ? 8e7 : ph === "entry" ? 2.5e7 : 6e6;
      const d = big * zoom, a = 0.6 + az, e = 0.5 + el;
      const ctr = ph === "coast" || ph === "tei" && t > M.tei[1] + 1800 ? pm.clone().add(V(0, 0, 0)).multiplyScalar(0.5).sub(O) : nearMoon ? pm.clone().sub(O) : V(-O.x, -O.y, -O.z);
      const n = V(0, 0, 1);
      const ref = V(1, 0, 0);
      camera.position.copy(ctr.clone().add(ref.clone().multiplyScalar(Math.cos(e) * Math.cos(a) * d)).add(n.clone().cross(ref).multiplyScalar(Math.cos(e) * Math.sin(a) * d)).add(n.clone().multiplyScalar(Math.sin(e) * d)));
      camera.up.copy(n); camera.fov = 45; camera.lookAt(ctr);
    } else if (preset === "Columbia's window" && (ph === "rendezvous" || ph === "ascent" || ph === "surface" || ph === "docked")) {
      const eye = csm.position.clone().add(V(0.6, 2.0, -1.2).applyMatrix4(Rc));
      const target = ascent.position.clone();
      camera.position.copy(eye);
      camera.up.copy(V(0, 1, 0).applyMatrix4(Rc));
      const dist = eye.distanceTo(target);
      camera.fov = Math.max(1, Math.min(60, 2 * Math.atan(18 / dist) / RAD)) / Math.sqrt(zoom);
      camera.lookAt(target);
    } else {
      const base = ph === "entry" && t >= M.drogue ? 140 : ph === "surface" ? 40 : 32;
      const d = base * zoom, a = 2.5 + az, e = 0.28 + el;
      const look = focusPos.clone().add(upRef.clone().multiplyScalar(ph === "entry" && t >= M.drogue ? 25 : 2));
      let off = fwd.clone().multiplyScalar(Math.cos(e) * Math.cos(a)).add(side.clone().multiplyScalar(Math.cos(e) * Math.sin(a)))
        .add(upRef.clone().multiplyScalar(Math.sin(e)));
      if (ph === "coast" || (ph === "tei" && t > M.tei[1] + 600) || (ph === "entry" && t < M.ei - 120)) {
        // deep space: look from the sunlit side, a little off the Sun line, with Earth in the background
        const toE = O.clone().negate().normalize(), s0 = sd.clone().normalize();
        const perp = new THREE.Vector3().crossVectors(s0, toE).normalize();
        off = s0.multiplyScalar(0.75).add(toE.multiplyScalar(-0.45)).add(perp.multiplyScalar(0.45)).normalize()
          .applyAxisAngle(toE.clone().negate().normalize(), az).applyAxisAngle(perp, el);
      }
      camera.position.copy(off.multiplyScalar(d).add(look));
      camera.fov = 50; camera.lookAt(look);
    }
    camera.aspect = W / H; camera.updateProjectionMatrix();
    dirty = true;
  }

  function csmAttitude(t) {
    const p = csmPos(t);
    const ph = phaseOf(t);
    const nearMoon = t < M.tei[1] + 3 * 3600;
    const ctr = nearMoon ? moonPos(t) : V(0, 0, 0);
    const up = p.clone().sub(ctr).normalize();
    const v = vel(csmPos, t, 2).sub(nearMoon ? vel(moonPos, t, 30) : V(0, 0, 0));
    if (ph === "rendezvous" || ph === "ascent" || ph === "surface") {
      // nose toward Eagle (Collins tracked it with the sextant and VHF ranging)
      const toL = lmPos(t).sub(p);
      return basis(toL.lengthSq() > 1 ? toL : v, up.clone().negate());
    }
    if (ph === "docked" || ph === "csm") return basis(v.clone().negate(), up.clone().negate());   // SPS forward, heads down
    if (t >= M.tei[0] - 60 && t <= M.tei[1] + 30) return basis(v.clone().negate(), up.clone().negate());   // SPS thrust along velocity: nose aft
    if (t >= M.ei - 60) {
      // entry: heat shield forward, trimmed about 20 deg, rolled by the bank angle
      const r = tabAt(entryRows, t, 8), bk = (r[7] || 0) * RAD;
      const vv = vel(csmPos, t, 1).normalize();
      const upp = up.clone().sub(vv.clone().multiplyScalar(up.dot(vv))).normalize();
      const lift = upp.clone().applyAxisAngle(vv, -bk);
      const axis = vv.clone().negate().applyAxisAngle(new THREE.Vector3().crossVectors(vv, lift).normalize(), -20 * RAD);
      if (t > M.drogue) return basis(up, V(1, 0, 0));
      return basis(axis.clone().negate(), lift);
    }
    // coast: passive thermal control, a slow roll (about 3 revolutions an hour) with the long axis across the Sun line
    const s = sunDir(t), a = new THREE.Vector3().crossVectors(s, V(0, 0, 1)).normalize();
    const ang = (t - M.tei[1]) * 2 * Math.PI / 1200;
    return basis(a, s.clone().applyAxisAngle(a, ang));
  }

  let raf = null;
  const loop = () => { if (dirty) { renderer.render(scene, camera); dirty = false; } raf = requestAnimationFrame(loop); };
  raf = requestAnimationFrame(loop);
  const setPreset = name => { preset = name; az = 0; el = 0; zoom = 1; update(T); };
  return {canvas: el$, update, setPreset, presets, info: () => info, phaseOf,
    dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); }};
}
