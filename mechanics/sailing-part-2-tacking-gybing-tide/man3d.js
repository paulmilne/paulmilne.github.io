// 3D view of the J/109 tacking and gybing, for the sailing post Part 2.
// World frame: y up; the true wind blows from +z towards -z (upwind is +z). The boat moves and
// turns in this frame; its heading, heel, leeway and position come from the manoeuvre
// simulation. Boat geometry as in Part 1 (metres). Sail shapes and the timing of the sails
// crossing are drawn to suggest the manoeuvre; the forces and motion are the model's.

export function createScene(THREE, M, opts = {}) {
  const W0 = opts.width || 640, H0 = opts.height || 420;
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(2, (typeof devicePixelRatio !== "undefined" ? devicePixelRatio : 1)));
  renderer.setSize(W0, H0);
  const canvas = renderer.domElement;
  canvas.style.width = "100%"; canvas.style.height = "auto"; canvas.style.display = "block";
  canvas.style.borderRadius = "6px"; canvas.style.touchAction = "none";
  const scene = new THREE.Scene();
  const skyCol = new THREE.Color(0xbcd4e6);
  scene.background = skyCol;
  scene.fog = new THREE.Fog(skyCol, 150, 600);
  const camera = new THREE.PerspectiveCamera(42, W0 / H0, 0.1, 3000);
  scene.add(new THREE.HemisphereLight(0xeaf4ff, 0x2b4a5e, 0.75));
  const sunL = new THREE.DirectionalLight(0xffffff, 0.9);
  sunL.position.set(-30, 60, 40);
  scene.add(sunL);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const RAD = Math.PI / 180;

  // ---- sea: a large world-fixed plane, and a wave patch that follows the boat --------------
  const mkTex = (bg, line, rep) => {
    const c = document.createElement("canvas"); c.width = c.height = 128;
    const g = c.getContext("2d");
    g.fillStyle = bg; g.fillRect(0, 0, 128, 128);
    g.strokeStyle = line; g.lineWidth = 2;
    g.beginPath(); g.moveTo(0, 64); g.lineTo(128, 64); g.moveTo(64, 0); g.lineTo(64, 128); g.stroke();
    g.fillStyle = line;
    for (let i = 0; i < 40; i++) g.fillRect((i * 53) % 128, (i * 97) % 128, 6, 1.5);
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rep, rep);
    return t;
  };
  const SEA = 3000;
  const water = new THREE.Mesh(new THREE.PlaneGeometry(SEA, SEA),
    new THREE.MeshPhongMaterial({color: 0xffffff, map: mkTex("#2f6b8f", "rgba(255,255,255,0.09)", SEA / 10), transparent: true, opacity: 0.66, shininess: 60, depthWrite: false}));
  water.rotation.x = -Math.PI / 2; water.position.y = -0.03;
  scene.add(water);
  const WP = 200, WN = 140;
  const wTex = mkTex("#ffffff", "rgba(0,0,0,0.07)", WP / 10);
  const wgeo = new THREE.PlaneGeometry(WP, WP, WN, WN);
  wgeo.rotateX(-Math.PI / 2);
  const wpos = wgeo.attributes.position, nV = wpos.count;
  const wx = new Float32Array(nV), wz = new Float32Array(nV), wfade = new Float32Array(nV);
  for (let i = 0; i < nV; i++) {
    wx[i] = wpos.getX(i); wz[i] = wpos.getZ(i);
    const r = Math.hypot(wx[i], wz[i]) / (WP / 2);
    wfade[i] = r >= 1 ? 0 : r < 0.6 ? 1 : 0.5 + 0.5 * Math.cos(Math.PI * (r - 0.6) / 0.4);
  }
  const wcol = new Float32Array(nV * 3);
  wgeo.setAttribute("color", new THREE.BufferAttribute(wcol, 3));
  const waves = new THREE.Mesh(wgeo, new THREE.MeshPhongMaterial({color: 0xffffff, map: wTex, vertexColors: true, shininess: 70, specular: 0x557799, transparent: true, opacity: 0.66, depthWrite: false}));
  scene.add(waves);
  const G = 9.81, FETCH = 20000;
  let trains = [], whitecap = 0;
  function setSea(tws) {
    const U = Math.max(0.5, tws * 0.514444), xf = G * FETCH / (U * U);
    const hs = Math.min(1.6e-3 * Math.sqrt(xf) * U * U / G, 0.21 * U * U / G);
    const tp = Math.min(0.286 * Math.cbrt(xf) * U / G, 7.14 * U / G);
    const lp = G * tp * tp / (2 * Math.PI);
    const dir0 = -Math.PI / 2;                    // waves run towards -z (downwind)
    const spec = [[1.0, 0, 0.55, 0.3], [0.62, 0.45, 0.33, 2.1], [0.41, -0.38, 0.24, 4.0], [0.26, 0.85, 0.13, 5.3], [0.17, -0.7, 0.08, 1.1]];
    trains = spec.map(([lf, dd, af, ph]) => {
      const lam = Math.max(0.6, lf * lp), k = 2 * Math.PI / lam, d = dir0 + dd;
      return {kx: k * Math.cos(d), kz: k * Math.sin(d), w: Math.sqrt(G * k), a: Math.min(af * hs / 2, lam / 18), ph};
    });
    whitecap = tws >= 22 ? 1 : tws >= 17 ? 0.75 : tws >= 11 ? 0.45 : 0;
  }
  const height = (X, Z, t) => { let h = 0; for (const q of trains) h += q.a * Math.cos(q.kx * X + q.kz * Z - q.w * t + q.ph); return h; };
  const base = new THREE.Color(0x2f6b8f), foam = new THREE.Color(0xf4f8fb), tmpC = new THREE.Color();
  function updateWaves(t, X0, Z0) {
    waves.position.set(X0, 0, Z0);
    wTex.offset.set(X0 / 10, -Z0 / 10);
    const am = trains.reduce((s, q) => s + q.a, 0) || 1;
    for (let i = 0; i < nV; i++) {
      const h = height(X0 + wx[i], Z0 + wz[i], t) * wfade[i];
      wpos.setY(i, h);
      const c = whitecap * Math.max(0, Math.min(1, (h / am - 0.62) * 4));
      tmpC.copy(base).lerp(foam, c);
      wcol[3 * i] = tmpC.r; wcol[3 * i + 1] = tmpC.g; wcol[3 * i + 2] = tmpC.b;
    }
    wpos.needsUpdate = true; wgeo.attributes.color.needsUpdate = true;
    wgeo.computeVertexNormals();
  }

  // ---- compass rose on the water ------------------------------------------------------------
  const ROSE_R = 19;
  const roseTex = (() => {
    const N = 1024, c = document.createElement("canvas"); c.width = c.height = N;
    const g = c.getContext("2d"), cx = N / 2, R = N * 0.47;
    g.clearRect(0, 0, N, N);
    g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 6;
    g.beginPath(); g.arc(cx, cx, R, 0, 2 * Math.PI); g.stroke();
    g.lineWidth = 3; g.beginPath(); g.arc(cx, cx, R * 0.86, 0, 2 * Math.PI); g.stroke();
    for (let b = 0; b < 360; b += 5) {
      const a = b * Math.PI / 180, len = b % 30 === 0 ? 0.1 : b % 10 === 0 ? 0.065 : 0.035;
      g.lineWidth = b % 30 === 0 ? 5 : 3;
      g.beginPath();
      g.moveTo(cx + R * Math.sin(a), cx - R * Math.cos(a));
      g.lineTo(cx + R * (1 - len) * Math.sin(a), cx - R * (1 - len) * Math.cos(a));
      g.stroke();
    }
    g.textAlign = "center"; g.textBaseline = "middle";
    for (let b = 0; b < 360; b += 30) {
      const a = b * Math.PI / 180, r = R * 0.76;
      const lab = {0: "N", 90: "E", 180: "S", 270: "W"}[b] ?? String(b).padStart(3, "0");
      g.font = lab.length === 1 ? "bold 64px sans-serif" : "bold 38px sans-serif";
      g.fillStyle = b === 0 ? "#ff5a4f" : "rgba(255,255,255,0.95)";
      g.save(); g.translate(cx + r * Math.sin(a), cx - r * Math.cos(a)); g.rotate(a); g.fillText(lab, 0, 0); g.restore();
    }
    const t = new THREE.CanvasTexture(c); t.anisotropy = 4;
    return t;
  })();
  const rose = new THREE.Mesh(new THREE.PlaneGeometry(2 * ROSE_R / 0.94, 2 * ROSE_R / 0.94),
    new THREE.MeshBasicMaterial({map: roseTex, transparent: true, depthWrite: false, opacity: 0.9}));
  rose.geometry.rotateX(-Math.PI / 2);
  rose.renderOrder = 2;
  scene.add(rose);

  // ---- the boat -------------------------------------------------------------------------
  const boat = new THREE.Group();          // origin: waterline, mid-length, on the centreline
  scene.add(boat);
  const L = 10.75, B = M.beam, FB = 1.15, T = M.tc;
  const hullMat = new THREE.MeshPhongMaterial({color: 0xf4f4f0, side: THREE.DoubleSide, shininess: 40});
  const deckMat = new THREE.MeshPhongMaterial({color: 0xd9d2c3, side: THREE.DoubleSide});
  const darkMat = new THREE.MeshPhongMaterial({color: 0x30343a});
  const stripeMat = new THREE.MeshPhongMaterial({color: 0x1d3557, side: THREE.DoubleSide});

  // Hull: sections along x; half-breadth and depth fall away towards the bow and the transom.
  const NS = 28, NA = 14;
  const halfB = u => { // u: 0 at the transom, 1 at the bow
    const s = Math.max(0, 1 - Math.pow(Math.max(0, u - 0.42) / 0.58, 1.9));
    return B / 2 * (0.82 + 0.18 * Math.sin(Math.min(1, u / 0.42) * Math.PI / 2)) * Math.sqrt(s);
  };
  const depth = u => T * Math.pow(Math.sin(Math.min(1, Math.max(0, (u - 0.03) / 0.9)) * Math.PI), 0.7);
  const sheer = u => FB + 0.25 * Math.pow(u - 0.35, 2);
  const pos = [], idx = [];
  for (let i = 0; i <= NS; i++) {
    const u = i / NS, x = -L / 2 + u * L, b = halfB(u), d = depth(u), s = sheer(u) - 0.05 * (1 - u);
    for (let j = 0; j <= NA; j++) {
      const a = -Math.PI / 2 + Math.PI * j / NA;              // -90 (port sheer) .. +90 (starboard sheer)
      const c = Math.cos(a), sgn = Math.sign(Math.sin(a)) || 0;
      const zz = b * Math.pow(Math.abs(Math.sin(a)), 0.55) * sgn;
      const yy = -d * Math.pow(c, 1.4) + (s + 0) * Math.pow(Math.abs(Math.sin(a)), 6);
      pos.push(x, yy, zz);
    }
  }
  for (let i = 0; i < NS; i++) for (let j = 0; j < NA; j++) {
    const a = i * (NA + 1) + j, b2 = a + NA + 1;
    idx.push(a, b2, a + 1, a + 1, b2, b2 + 1);
  }
  const hg = new THREE.BufferGeometry();
  hg.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  hg.setIndex(idx); hg.computeVertexNormals();
  boat.add(new THREE.Mesh(hg, hullMat));
  // deck
  const deckShape = new THREE.Shape();
  deckShape.moveTo(-L / 2, halfB(0));
  for (let i = 1; i <= NS; i++) { const u = i / NS; deckShape.lineTo(-L / 2 + u * L, halfB(u)); }
  for (let i = NS; i >= 0; i--) { const u = i / NS; deckShape.lineTo(-L / 2 + u * L, -halfB(u)); }
  const deck = new THREE.Mesh(new THREE.ShapeGeometry(deckShape), deckMat);
  deck.rotation.x = Math.PI / 2; deck.position.y = FB + 0.02;
  boat.add(deck);
  // transom
  const tr = new THREE.Mesh(new THREE.PlaneGeometry(halfB(0) * 2, FB + 0.1), hullMat);
  tr.rotation.y = Math.PI / 2; tr.position.set(-L / 2, FB / 2, 0);
  boat.add(tr);
  // boot stripe
  const stripe = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 0.08, 40, 1, true), stripeMat);
  stripe.scale.set(L / 2 * 0.93, 1, B / 2 * 0.98); stripe.position.y = 0.12;
  boat.add(stripe);

  // Keel: trapezoidal fin and a lead bulb; rudder.
  const foil = (root, tip, span, sweep, thick) => {
    const g = new THREE.BufferGeometry(), p = [], ix = [];
    const pts = [[0, 0], [root, 0], [sweep + tip, -span], [sweep, -span]];
    for (const [x, y] of pts) p.push(x, y, thick / 2);
    for (const [x, y] of pts) p.push(x, y, -thick / 2);
    ix.push(0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0);
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); g.setIndex(ix); g.computeVertexNormals();
    return g;
  };
  const [kr, kt, ks] = M.keel, [rr, rt, rs] = M.rudder;
  const keel = new THREE.Mesh(foil(kr, kt, ks, 0.32, 0.11), new THREE.MeshPhongMaterial({color: 0x3d4a5c}));
  keel.position.set(-kr / 2 + 0.3, -T + 0.05, 0);
  boat.add(keel);
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12), darkMat);
  bulb.scale.set(1.25, 0.2, 0.22); bulb.position.set(0.15, -M.tmax + 0.18, 0);
  boat.add(bulb);
  // Rudder: a spade hung on a stock about 1.2 m forward of the transom, turned by the wheel.
  const rudderX = -L / 2 + 1.2;
  const rudderPivot = new THREE.Group(); rudderPivot.position.set(rudderX, 0, 0); boat.add(rudderPivot);
  const rudderMat = new THREE.MeshPhongMaterial({color: 0xc0392b});
  const rudder = new THREE.Mesh(foil(rr, rt, rs, 0.1, 0.06), rudderMat);
  rudder.position.set(-0.75 * rr, -0.22, 0);         // stock a quarter of the chord back from the leading edge
  rudderPivot.add(rudder);
  const stock = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.5, 8), darkMat);
  stock.position.set(0, 0.5, 0); rudderPivot.add(stock);
  // Wheel in the cockpit, aft of the helmsman's position, facing forward.
  const wheel = new THREE.Group(); wheel.position.set(rudderX + 0.6, FB + 0.75, 0); boat.add(wheel);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.6, 0.025, 8, 40), rigMat0());
  rim.rotation.y = Math.PI / 2; wheel.add(rim);
  for (let k = 0; k < 6; k++) {
    const sp = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1.2, 6), rigMat0());
    sp.rotation.x = k * Math.PI / 6; wheel.add(sp);
  }
  const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.75, 10), darkMat);
  pedestal.position.set(rudderX + 0.75, FB + 0.38, 0); boat.add(pedestal);
  function rigMat0() { return new THREE.MeshPhongMaterial({color: 0x8a9099}); }

  // Rig: mast (stepped about 4.2 m from the bow), boom, forestay, bowsprit.
  const mastX = L / 2 - 4.35, deckY = M.hbi;
  const mastTop = deckY + M.I + 0.5;
  const rigMat = new THREE.MeshPhongMaterial({color: 0x8a9099});
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, mastTop - deckY, 10), rigMat);
  mast.position.set(mastX, (mastTop + deckY) / 2, 0);
  boat.add(mast);
  const gooseY = deckY + M.bad;
  const boomPivot = new THREE.Group(); boomPivot.position.set(mastX - 0.1, gooseY, 0); boat.add(boomPivot);
  const boomMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, M.E, 8), rigMat);
  boomMesh.rotation.z = Math.PI / 2; boomMesh.position.x = -M.E / 2; boomPivot.add(boomMesh);
  const stayMat = new THREE.LineBasicMaterial({color: 0x555555});
  const line = (pts, mat = stayMat) => { const g = new THREE.BufferGeometry().setFromPoints(pts); const l = new THREE.Line(g, mat); boat.add(l); return l; };
  const bowX = L / 2 - 0.15;
  line([V(bowX, FB, 0), V(mastX, deckY + M.I, 0)]);
  line([V(mastX, deckY + M.I * 0.97, 0), V(mastX - 0.2, FB, B / 2 - 0.1)]);
  line([V(mastX, deckY + M.I * 0.97, 0), V(mastX - 0.2, FB, -B / 2 + 0.1)]);
  line([V(mastX, mastTop, 0), V(-L / 2 + 0.2, FB, 0)]);
  const sprit = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.6, 8), rigMat);
  sprit.rotation.z = Math.PI / 2; sprit.position.set(bowX + 0.7, FB + 0.05, 0); boat.add(sprit);

  // Sails: cambered surfaces between luff and leech, rebuilt each update.
  const sailMat = c => new THREE.MeshPhongMaterial({color: c, side: THREE.DoubleSide, transparent: true, opacity: 0.93, shininess: 10});
  const mkSail = (mat) => { const m = new THREE.Mesh(new THREE.BufferGeometry(), mat); boat.add(m); return m; };
  const mainS = mkSail(sailMat(0xfaf8f2)), jibS = mkSail(sailMat(0xf1efe8)), kiteS = mkSail(sailMat(0xe76f51));
  // A sail between a luff line (a0 -> a1) and leech/clew points, with camber pushed to leeward.
  function sailGeom(mesh, luff0, luff1, clew, head, camber, side, nu = 10, nv = 8) {
    // luff0: tack, luff1: head on the luff; clew: clew; head: head (same as luff1 normally)
    const p = [], ix = [];
    for (let i = 0; i <= nv; i++) {
      const v = i / nv;
      const a = luff0.clone().lerp(luff1, v);              // point on the luff
      const b = clew.clone().lerp(head, v);                // point on the leech
      const chord = b.clone().sub(a), cl = chord.length();
      const n = side.clone().multiplyScalar(camber * cl);
      for (let j = 0; j <= nu; j++) {
        const u = j / nu, bulge = 4 * u * (1 - u) * (1 - 0.3 * u);
        p.push(...a.clone().add(chord.clone().multiplyScalar(u)).add(n.clone().multiplyScalar(bulge)).toArray());
      }
    }
    for (let i = 0; i < nv; i++) for (let j = 0; j < nu; j++) {
      const a = i * (nu + 1) + j, b = a + nu + 1;
      ix.push(a, b, a + 1, a + 1, b, b + 1);
    }
    mesh.geometry.dispose();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); g.setIndex(ix); g.computeVertexNormals();
    mesh.geometry = g;
  }

  // ---- trail, VMG line and markers ---------------------------------------------------------
  const trailGeo = new THREE.BufferGeometry();
  const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.8}));
  scene.add(trail);
  const refMat = new THREE.MeshBasicMaterial({color: 0xe9c46a, transparent: true, opacity: 0.85});
  const refLine = new THREE.Mesh(new THREE.BoxGeometry(160, 0.08, 0.35), refMat);
  scene.add(refLine);
  const windArrow = new THREE.ArrowHelper(V(0, 0, -1), V(0, 0, 0), 9, 0x1d4ed8, 1.4, 0.8);
  scene.add(windArrow);

  // ---- forces ------------------------------------------------------------------------------
  const arrows = {};
  const mkArrow = (name, color) => {
    const a = new THREE.ArrowHelper(V(1, 0, 0), V(0, 0, 0), 1, color, 0.6, 0.35);
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 1, 8), new THREE.MeshBasicMaterial({color}));
    a.add(shaft); a.userData.shaft = shaft;
    scene.add(a); arrows[name] = a; return a;
  };
  const setArrow = (a, origin, vec, scale) => {
    const len = vec.length() * scale;
    a.visible = len > 0.05;
    if (!a.visible) return;
    a.position.copy(origin);
    a.setDirection(vec.clone().normalize());
    const head = Math.min(0.9, 0.35 * len + 0.15);
    a.setLength(len, head, head * 0.55);
    const sh = a.userData.shaft;
    sh.scale.set(1, Math.max(0.01, len - head), 1);
    sh.position.set(0, (len - head) / 2, 0);
  };
  const C = {aero: 0xe63946, keel: 0x457b9d, res: 0x6d597a};
  for (const [k, c] of Object.entries(C)) mkArrow(k, c);

  // ---- camera ------------------------------------------------------------------------------
  const presets = ["Chase", "Downwind", "Above", "Bow"];
  let preset = "Chase", az = 0, el = 0, zoom = 1, drag = null;
  canvas.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    az -= (e.clientX - drag[0]) * 0.008; el = Math.max(-0.5, Math.min(1.3, el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; render();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.3, Math.min(4, zoom * Math.exp(e.deltaY * 0.001))); render(); }, {passive: false});
  let camYaw = null;
  function placeCamera() {
    const p = boat.position;
    // Chase follows the boat's heading slowly; Downwind and Above are fixed to the wind, so the turn shows.
    const yaw = boat.rotation.y;
    if (camYaw === null) camYaw = yaw;
    camYaw += Math.atan2(Math.sin(yaw - camYaw), Math.cos(yaw - camYaw)) * 0.06;
    const base = {Chase: [camYaw + Math.PI + 0.3, 0.32, 34], Downwind: [Math.PI / 2, 0.22, 46], Above: [Math.PI / 2, 1.45, 70], Bow: [camYaw + 0.5, 0.22, 30]}[preset];
    const a = base[0] + az, e = Math.min(1.52, base[1] + el), d = base[2] * zoom;
    const tgt = V(p.x, 4, p.z);
    // yaw convention: local +x of the boat is (cos yaw, 0, -sin yaw); put the camera along that direction rotated by a
    camera.position.set(tgt.x + d * Math.cos(e) * Math.cos(a), tgt.y + d * Math.sin(e), tgt.z - d * Math.cos(e) * Math.sin(a));
    camera.up.set(0, 1, 0);
    camera.lookAt(tgt);
  }
  function render() { placeCamera(); renderer.render(scene, camera); }
  let raf = null, S = null, simT = 0;
  const loop = () => { if (S) { updateWaves(simT, boat.position.x, boat.position.z); } render(); raf = requestAnimationFrame(loop); };
  raf = requestAnimationFrame(loop);

  // ---- update from a simulation sample -------------------------------------------------------
  // s: {t, twa (heading, signed: + wind over starboard), heel (+ to port), leeway, awa (signed), aws,
  //     lift, drag, keel (+ to starboard), rtot, x, z (world), fill, kite, cross, kind, tws}, trailPts: [[x,z],...], refZ
  let lastTws = null;
  const sgnf = x => (x >= 0 ? 1 : -1);
  function update(s, trailPts, refZ, showForces = true) {
    S = s; simT = s.t;
    if (s.tws !== lastTws) { setSea(s.tws); lastTws = s.tws; }
    const heel = s.heel * RAD;
    // yaw so that the wind (from +z) arrives at the signed angle twa off the bow
    const yaw = s.twa * RAD - Math.PI / 2;
    const hb = height(s.x + 4 * Math.cos(yaw), s.z - 4 * Math.sin(yaw), simT) - height(s.x - 4 * Math.cos(yaw), s.z + 4 * Math.sin(yaw), simT);
    boat.rotation.order = "YXZ";
    boat.rotation.set(-heel, yaw, Math.atan2(hb, 8) * 0.6);
    boat.position.set(s.x, 0.6 * height(s.x, s.z, simT), s.z);

    // Compass: the wind comes from +z, whose bearing is windFrom; bearings increase clockwise seen from above.
    rose.rotation.y = ((s.windFrom ?? 225) + 180) * RAD;
    rose.position.set(s.x, 0.45, s.z);
    // Sails, in the boat's frame. side: -1 sails to port (wind over starboard), +1 to starboard.
    const awa = s.awa, side = awa >= 0 ? -1 : 1, aa = Math.abs(awa);
    const sideV = V(0, 0, side);
    const dt = s.cross == null ? -1e9 : s.t - s.cross;   // time since the wind crossed the bow or stern
    // boom: on a gybe it swings across in about a second; otherwise it follows the apparent wind
    let boom = Math.max(2, Math.min(82, aa - 24));
    let boomSide = side;
    if (s.kind === "gybe" && s.cross != null && Math.abs(dt) < 0.8) {
      const oldS = dt < 0 ? side : -side, newS = -oldS;
      const f = (dt + 0.8) / 1.6, sm = f * f * (3 - 2 * f);  // 0 -> 1 across the swing
      const ang = oldS * boom * (1 - sm) + newS * boom * sm;   // signed, from the old side to the new
      boom = Math.max(1, Math.abs(ang)); boomSide = sgnf(ang);
    }
    boomPivot.rotation.y = boomSide * boom * RAD;
    const bA = boom * RAD;
    const reefH = s.reef || 1;
    const tackM = V(mastX - 0.1, gooseY, 0), headM = V(mastX, gooseY + M.P * reefH, 0);
    const clewM = V(mastX - 0.1 - M.E * Math.cos(bA), gooseY, boomSide * M.E * Math.sin(bA));
    const headLeech = V(mastX - 0.5 * reefH, gooseY + M.P * reefH, boomSide * 0.4 * Math.sin(bA));
    const luffing = s.kind === "tack" ? Math.max(0, 1 - s.fill) : 0;
    sailGeom(mainS, tackM, headM, clewM, headLeech, (0.07 + 0.07 * (s.flat || 1)) * (1 - 0.8 * luffing), V(0, 0, boomSide));
    jibS.visible = !s.kite; kiteS.visible = !!s.kite;
    if (!s.kite) {
      // the jib stays on the old side ("backed") for a moment after the bow crosses the wind, then is sheeted across
      let jSide = side, jAng = Math.max(7, Math.min(60, aa - 18));
      if (s.kind === "tack" && s.cross != null && dt >= 0 && dt < 2.0) {
        const f = dt / 2.0, sm = f * f * (3 - 2 * f);
        const ang = -side * 8 * (1 - sm) + side * jAng * sm;  // backed on the old side, then across to the new
        jSide = sgnf(ang); jAng = Math.max(1, Math.abs(ang));
      }
      const tackJ = V(bowX, FB + 0.15, 0), headJ = V(mastX + 0.25, deckY + M.I * 0.97, 0);
      const footL = 4.0;
      const clewJ = V(bowX - footL * Math.cos(jAng * RAD), FB + 0.35, jSide * footL * Math.sin(jAng * RAD));
      sailGeom(jibS, tackJ, headJ, clewJ, headJ, (0.09 + 0.04) * (1 - 0.85 * luffing), V(0, 0, jSide));
    } else {
      let kSide = side, kAng = Math.max(25, Math.min(95, aa - 25)), coll = 1;
      if (s.kind === "gybe" && s.cross != null && Math.abs(dt) < 1.6) {
        const oldS = dt < 0 ? side : -side, newS = -oldS;
        const f = (dt + 1.6) / 3.2, sm = f * f * (3 - 2 * f);
        const ang = oldS * kAng * (1 - sm) + newS * kAng * sm;
        kSide = sgnf(ang); kAng = Math.max(4, Math.abs(ang)); coll = 0.35 + 0.65 * Math.abs(2 * sm - 1);
      }
      const tackK = V(bowX + 1.45, FB + 0.4, 0), headK = V(mastX + 0.3, deckY + M.I * 0.98, 0);
      const foot = 6.6;
      const clewK = V(bowX + 1.4 - foot * Math.cos(kAng * RAD), FB + 1.4, kSide * (foot * Math.sin(kAng * RAD) + 0.6));
      sailGeom(kiteS, tackK, headK, clewK, headK, 0.22 * coll, V(Math.sin(kAng * RAD) * 0.6, 0.15, kSide).normalize(), 12, 10);
    }

    // Forces (world): sail force from lift and drag about the apparent wind; keel and resistance.
    boat.updateMatrixWorld(true);
    const Ryaw = new THREE.Matrix4().makeRotationY(yaw);
    const a = awa * RAD;
    const awDir = V(-Math.cos(a), 0, -Math.sin(a));                    // where the apparent wind goes, boat frame
    const liftDir = V(Math.sin(a), 0, -Math.cos(a)).multiplyScalar(sgnf(awa));
    const aero = liftDir.multiplyScalar(s.lift).add(awDir.multiplyScalar(s.drag)).applyMatrix4(Ryaw);
    const sc = 7 / 6000;
    const ce = V(mastX - 1.6, 7, 0).applyMatrix4(boat.matrixWorld);
    const clr = V(0.1, -0.9, 0).applyMatrix4(boat.matrixWorld);
    setArrow(arrows.aero, ce, showForces ? aero : V(0, 0, 0), sc);
    setArrow(arrows.keel, clr, showForces ? V(0, 0, s.keel).applyMatrix4(Ryaw) : V(0, 0, 0), sc);
    setArrow(arrows.res, V(0, 0.2, 0).add(boat.position), showForces ? V(-s.rtot, 0, 0).applyMatrix4(Ryaw) : V(0, 0, 0), sc);
    windArrow.position.set(s.x + 14, 15, s.z + 10);

    // Trail and the reference line (where a boat at its steady VMG would be).
    if (trailPts) trailGeo.setFromPoints(trailPts.map(([x, z]) => V(x, 0.15, z)));
    refLine.visible = refZ != null;
    if (refZ != null) refLine.position.set(s.x, 0.1, refZ);
    render();
  }
  function setPreset(p) { preset = p; az = 0; el = 0; zoom = 1; camYaw = null; render(); }
  return {canvas, update, setPreset, presets, dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); }};
}
