// CGI view of Apollo 11 for the Part 2 player: Columbia and Eagle docked, from TLI to lunar orbit.
// Built from primitives at real dimensions (metres). World frame: geocentric, mean equator and
// equinox of 16 July 1969 (z = north pole), with a floating origin at the spacecraft.

const OMEGA = 7.2921158553e-5;
const PTC_RATE = 0.3 * Math.PI / 180;        // PTC roll rate, 0.3 deg/s (about 3 turns an hour)
const EPS = 23.4423 * Math.PI / 180;          // obliquity of the ecliptic, 1969

// Approximate maria: [name, lat, lon, radius in degrees]. Used for the texture and the globe panel.
export const MARIA = [
  ["Oceanus Procellarum", 20, -57, 30], ["Oceanus Procellarum", -2, -45, 14], ["Mare Imbrium", 33, -16, 18],
  ["Mare Frigoris", 56, -10, 6], ["Mare Frigoris", 57, 10, 6], ["Mare Frigoris", 55, 28, 5],
  ["Mare Serenitatis", 28, 17.5, 10.5], ["Mare Tranquillitatis", 8.5, 31, 12], ["Mare Tranquillitatis", 13, 25, 6],
  ["Mare Crisium", 17, 59, 8], ["Mare Fecunditatis", -7.8, 51.3, 10], ["Mare Nectaris", -15.2, 35.5, 5.5],
  ["Mare Nubium", -21, -17, 11], ["Mare Cognitum", -10, -23, 6], ["Mare Humorum", -24.4, -38.6, 6.5],
  ["Mare Vaporum", 13.3, 3.6, 4], ["Sinus Medii", 2, 1, 3.5], ["Mare Insularum", 7.5, -30.9, 7],
  ["Mare Smythii", 1.3, 87.5, 6], ["Mare Marginis", 13.3, 86.1, 5], ["Mare Moscoviense", 27.3, 147.9, 4.5],
  ["Mare Orientale", -19.4, -92.8, 5], ["Mare Australe", -38.9, 93, 8], ["Mare Ingenii", -33.7, 163.5, 4],
];

export function createScene(THREE, {data, land, d3, width, height}) {
  const W = width, H = height;
  const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H);
  renderer.outputEncoding = THREE.sRGBEncoding;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02040a);
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.5, 2e10);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);

  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  scene.add(sun, sun.target);
  scene.add(new THREE.AmbientLight(0x8899aa, 0.22));

  {
    const g = new THREE.BufferGeometry(), n = 3000, p = new Float32Array(n * 3);
    let seed = 11; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) {
      const z = 2 * rnd() - 1, a = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
      p.set([5e9 * r * Math.cos(a), 5e9 * r * Math.sin(a), 5e9 * z], 3 * i);
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({color: 0xffffff, size: 1.2, sizeAttenuation: false})));
  }

  // A lat/long sphere whose texture is an equirectangular canvas (lon -180..180, north up).
  const texturedSphere = (R, draw, nLat = 90, nLon = 180) => {
    const tw = R > 3e6 ? 2048 : 4096, th = tw / 2, cv = document.createElement("canvas");
    cv.width = tw; cv.height = th;
    const ctx = cv.getContext("2d");
    draw(ctx, d3.geoEquirectangular().scale(tw / (2 * Math.PI)).translate([tw / 2, th / 2]), tw, th);
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding; tex.anisotropy = 8;
    const pos = [], uv = [], idx = [];
    for (let i = 0; i <= nLat; i++) {
      const lat = -Math.PI / 2 + Math.PI * i / nLat;
      for (let j = 0; j <= nLon; j++) {
        const lon = -Math.PI + 2 * Math.PI * j / nLon;
        pos.push(R * Math.cos(lat) * Math.cos(lon), R * Math.cos(lat) * Math.sin(lon), R * Math.sin(lat));
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
    return new THREE.Mesh(g, new THREE.MeshStandardMaterial({map: tex, roughness: 0.95, metalness: 0}));
  };

  const RE = 6378166, FLAT = 1 / 298.3;
  const earth = new THREE.Group();
  earth.add(texturedSphere(RE, (ctx, proj, tw, th) => {
    ctx.fillStyle = "#123a63"; ctx.fillRect(0, 0, tw, th);
    const path = d3.geoPath(proj, ctx);
    if (land) { ctx.beginPath(); path(land); ctx.fillStyle = "#4a6b3d"; ctx.fill(); }
  }));
  earth.add(new THREE.Mesh(new THREE.SphereGeometry(RE * 1.015, 96, 48),
    new THREE.MeshBasicMaterial({color: 0x5aa0ff, transparent: true, opacity: 0.12, side: THREE.BackSide})));
  earth.scale.set(1, 1, 1 - FLAT);
  scene.add(earth);

  // The Moon: highland grey, darker maria, a scatter of craters. Body frame x towards mean Earth.
  const RM = 1738090;
  const site = data.meta.landing_site;
  const moon = texturedSphere(RM, (ctx, proj, tw, th) => {
    ctx.fillStyle = "#9b9a95"; ctx.fillRect(0, 0, tw, th);
    let seed = 3; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const path = d3.geoPath(proj, ctx);
    for (let k = 0; k < 900; k++) {          // mottled highlands
      const lat = Math.asin(2 * rnd() - 1) * 180 / Math.PI, lon = 360 * rnd() - 180;
      ctx.beginPath(); path(d3.geoCircle().center([lon, lat]).radius(1 + 6 * rnd() ** 2)());
      ctx.fillStyle = rnd() < 0.5 ? "rgba(255,255,250,0.05)" : "rgba(40,40,40,0.05)"; ctx.fill();
    }
    MARIA.forEach(([, lat, lon, r]) => {
      ctx.beginPath(); path(d3.geoCircle().center([lon, lat]).radius(r)());
      ctx.fillStyle = "rgba(62,64,70,0.78)"; ctx.fill();
    });
    for (let k = 0; k < 500; k++) {           // craters: bright rim, darker floor
      const lat = Math.asin(2 * rnd() - 1) * 180 / Math.PI, lon = 360 * rnd() - 180, r = 0.3 + 2.5 * rnd() ** 3;
      ctx.beginPath(); path(d3.geoCircle().center([lon, lat]).radius(r)());
      ctx.fillStyle = "rgba(30,30,30,0.18)"; ctx.fill();
      ctx.strokeStyle = "rgba(255,255,245,0.25)"; ctx.lineWidth = 1; ctx.stroke();
    }
    ctx.beginPath(); path(d3.geoCircle().center([site[1], site[0]]).radius(0.15)());
    ctx.fillStyle = "#e76f51"; ctx.fill();
  }, 120, 240);
  moon.matrixAutoUpdate = false;
  scene.add(moon);

  // ---- Columbia (CSM) with Eagle (LM) docked nose to nose. Local +y is the CSM's nose. ----
  const silver = new THREE.MeshStandardMaterial({color: 0xd8dade, roughness: 0.25, metalness: 0.85});
  const grey = new THREE.MeshStandardMaterial({color: 0x9aa0a6, roughness: 0.5, metalness: 0.3});
  const dark = new THREE.MeshStandardMaterial({color: 0x3a3d42, roughness: 0.6, metalness: 0.2});
  const gold = new THREE.MeshStandardMaterial({color: 0xc9a227, roughness: 0.35, metalness: 0.7});
  const engineMat = new THREE.MeshStandardMaterial({color: 0x2b2b2b, roughness: 0.4, metalness: 0.6, side: THREE.DoubleSide});
  const cyl = (r0, r1, h, mat, y, seg = 48) => {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, h, seg), mat); m.position.y = y + h / 2; return m;
  };
  const stack = new THREE.Group();
  const RS = 1.96;
  {
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 1.25, 2.9, 24, 1, true), engineMat);
    bell.position.y = -1.45; stack.add(bell);
    stack.add(cyl(RS, RS, 3.9, silver, 0));
    stack.add(cyl(RS, 0.45, 3.2, silver, 3.9));
    for (let k = 0; k < 4; k++) {
      const q = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.9, 0.5), grey), a = k * Math.PI / 2 + Math.PI / 4;
      q.position.set((RS + 0.2) * Math.cos(a), 2.6, (RS + 0.2) * Math.sin(a)); stack.add(q);
    }
    // High-gain antenna on its boom at the SM base.
    const boom = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 1.6, 6), grey);
    boom.position.set(RS + 0.6, 0.2, 0); boom.rotation.z = Math.PI / 2; stack.add(boom);
    [-0.35, 0.35].forEach(dz => [-0.35, 0.35].forEach(dy => {
      const dish = new THREE.Mesh(new THREE.CircleGeometry(0.4, 16), silver);
      dish.position.set(RS + 1.45, 0.2 + dy, dz); dish.rotation.y = Math.PI / 2; stack.add(dish);
    }));
    // Eagle, upside down on the CM's nose: ascent stage nearest Columbia.
    const lm = new THREE.Group();
    const ds = new THREE.Mesh(new THREE.CylinderGeometry(2.1, 2.1, 1.7, 8), gold); ds.position.y = 1.2; lm.add(ds);
    const as = new THREE.Mesh(new THREE.BoxGeometry(2.8, 2.3, 2.6), grey); as.position.y = 3.25; lm.add(as);
    const asTop = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.3, 0.6, 8), dark); asTop.position.y = 4.6; lm.add(asTop);
    const port = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.6, 0.4, 16), silver); port.position.y = 5.1; lm.add(port);
    for (let k = 0; k < 4; k++) {
      const a = Math.PI / 4 + k * Math.PI / 2;
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 6), silver);
      leg.position.set(2.0 * Math.cos(a), 1.0, 2.0 * Math.sin(a)); leg.rotation.z = 0.35 * Math.cos(a); leg.rotation.x = -0.35 * Math.sin(a);
      lm.add(leg);
    }
    lm.rotation.x = Math.PI;
    lm.position.y = 7.1 + 0.9 + 5.3;
    stack.add(lm);
  }
  const plumeMat = new THREE.MeshBasicMaterial({color: 0xccddff, transparent: true, opacity: 0.45, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide});
  const sps = new THREE.Mesh(new THREE.ConeGeometry(1.7, 16, 24, 1, true), plumeMat);
  sps.position.y = -2.9 - 8; stack.add(sps);
  const rcsMat = new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false});
  const rcs = [0, 1, 2, 3].map(k => {
    const m = new THREE.Mesh(new THREE.ConeGeometry(0.12, 1.2, 8, 1, true), rcsMat), a = k * Math.PI / 2 + Math.PI / 4;
    m.position.set((RS + 0.55) * Math.cos(a), 2.6, (RS + 0.55) * Math.sin(a));
    m.rotation.z = Math.PI / 2; m.rotation.y = -a + (k % 2 ? Math.PI : 0); stack.add(m); return m;
  });
  scene.add(stack);

  // ---- Trajectory and ephemeris lookups ----
  const tr = data.track.filter((d, i, a) => i === 0 || d[0] > a[i - 1][0]);
  const times = tr.map(d => d[0]);
  const rowAt = t => {
    const i = Math.max(0, Math.min(tr.length - 2, d3.bisectRight(times, t) - 1));
    const a = tr[i], b = tr[i + 1], f = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0])));
    return [1, 2, 3, 4, 5, 6].map(j => a[j] + f * (b[j] - a[j]));
  };
  const tab = (rows, t, n) => {
    const ts = rows._t ?? (rows._t = rows.map(d => d[0]));
    const i = Math.max(0, Math.min(rows.length - 2, d3.bisectRight(ts, t) - 1));
    const a = rows[i], b = rows[i + 1], f = (t - a[0]) / (b[0] - a[0]);
    return d3.range(1, n + 1).map(j => a[j] + f * (b[j] - a[j]));
  };
  // Positions (m): spacecraft geocentric, Moon geocentric (Moon centre = sc - sc_rel_moon).
  const scPos = t => { const r = rowAt(t); return {p: V(r[0], r[1], r[2]).multiplyScalar(1000), m: V(r[3], r[4], r[5]).multiplyScalar(1000)}; };
  const velocity = (t, rel) => {
    const a = scPos(t - 5), b = scPos(t + 5);
    return rel ? b.m.sub(a.m).multiplyScalar(0.1) : b.p.sub(a.p).multiplyScalar(0.1);
  };
  const lvlh = (r, v) => {
    const rh = r.clone().normalize();
    const x = v.clone().sub(rh.clone().multiplyScalar(v.dot(rh))).normalize();
    const z = rh.clone().negate();
    const y = new THREE.Vector3().crossVectors(z, x);
    return [x, y, z];
  };
  const fromLvlh = (d, [x, y, z]) => x.clone().multiplyScalar(d[0]).add(y.clone().multiplyScalar(d[1])).add(z.clone().multiplyScalar(d[2])).normalize();

  const B = data.burns;
  const burns = [
    {name: "Evasive burn", t0: B.evasive.tig, dur: B.evasive.dur, dir: [5.1, 0, 19.0], body: "earth"},
    {name: "Midcourse correction 2", t0: B.mcc.get - 1.56, dur: 3.13, dir: B.mcc.lvlh, body: "earth"},
    {name: "LOI-1", t0: B.loi1.tig, dur: B.loi1.dur, dir: B.loi1.dir, body: "moon", rate: B.loi1.pitch_rate},
    {name: "LOI-2", t0: B.loi2.tig, dur: B.loi2.dur, dir: B.loi2.dir, body: "moon"},
  ];
  const burnDir = (b, t) => {
    const tc = Math.max(b.t0, Math.min(b.t0 + b.dur, t));
    const s = scPos(tc), r = b.body === "moon" ? s.m : s.p, v = velocity(tc, b.body === "moon");
    let d = b.dir;
    if (b.rate) {
      const ang = b.rate * (tc - b.t0), c = Math.cos(ang), sn = Math.sin(ang);
      d = [c * d[0] + sn * d[2], d[1], -sn * d[0] + c * d[2]];
    }
    return fromLvlh(d, lvlh(r, v));
  };

  const eclN = V(0, -Math.sin(EPS), Math.cos(EPS));
  const quatFrom = (fwd, upHint, roll = 0) => {
    let z = upHint.clone().sub(fwd.clone().multiplyScalar(upHint.dot(fwd)));
    if (z.lengthSq() < 1e-8) z = V(1, 0, 0).sub(fwd.clone().multiplyScalar(fwd.x));
    z.normalize();
    const x = new THREE.Vector3().crossVectors(fwd, z).normalize();
    const q = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, fwd, z));
    if (roll) q.multiply(new THREE.Quaternion().setFromAxisAngle(V(0, 1, 0), roll));
    return q;
  };
  const LOI1_END = B.loi1.tig + B.loi1.dur;
  const PTC_ON = 11 * 3600, PTC_OFF = 74.5 * 3600, BLEND = 900;
  // Attitude mode and quaternion at time t.
  const modeAt = t => {
    const s = scPos(t);
    for (const b of burns) {
      if (t >= b.t0 - BLEND && t <= b.t0 + b.dur + 60) {
        const q = quatFrom(burnDir(b, t), s.p.clone().normalize());
        const burning = t >= b.t0 && t <= b.t0 + b.dur;
        return {q, label: burning ? `${b.name}: SPS burning` : `${b.name}: burn attitude`, burning, w: 1 - Math.max(0, Math.min(1, (b.t0 - t) / BLEND))};
      }
    }
    if (t > LOI1_END) {
      const v = velocity(t, true);
      return {q: quatFrom(v.clone().normalize(), s.m.clone().normalize().negate()), label: "Lunar orbit: nose forward, heads down", w: 1};
    }
    if (t > PTC_ON && t < PTC_OFF) {
      const q = quatFrom(eclN, V(...tab(data.sun, t, 3)), PTC_RATE * (t - PTC_ON));
      return {q, label: "Passive thermal control: 'barbecue' roll at 0.3°/s", w: 1};
    }
    const v = velocity(t, false);
    return {q: quatFrom(v.normalize(), s.p.clone().normalize()), label: t < PTC_ON ? "Coasting, docked" : "Preparing for lunar orbit insertion", w: 1};
  };

  // ---- Camera ----
  let preset = "Chase", az = 0, el = 0, zoom = 1;
  const presets = {
    "Chase": {},
    "Towards the Moon": {},
    "Towards Earth": {},
  };
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
  el$.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.2, Math.min(50, zoom * Math.exp(e.deltaY * 0.001))); update(T); }, {passive: false});

  let T = 0, dirty = true, prevQ = null, info = {label: ""};
  const mBody = new THREE.Matrix4();

  function update(t) {
    T = t;
    const s = scPos(t);
    const mode = modeAt(t);
    // Smooth hand-over into a burn attitude: slerp from the attitude the stack would otherwise have.
    let q = mode.q;
    if (mode.w < 1) {
      const t0 = burns.find(b => t >= b.t0 - BLEND && t < b.t0).t0 - BLEND - 1;
      q = modeAt(t0).q.clone().slerp(mode.q, mode.w * mode.w * (3 - 2 * mode.w));
    }
    stack.quaternion.copy(q);
    sps.visible = !!mode.burning;
    const fl = 0.9 + 0.1 * Math.sin(t * 41) * Math.sin(t * 17);
    sps.scale.set(fl, fl, fl);
    const ptcPulse = mode.label.startsWith("Passive") && (t % 120) < 1.5;
    rcs.forEach(r => { r.visible = ptcPulse || (mode.w < 1 && (t % 20) < 1); });
    info = {label: mode.label};

    // Floating origin at the spacecraft.
    const moonC = s.p.clone().sub(s.m);
    earth.position.copy(s.p.clone().negate());
    earth.rotation.set(0, 0, data.meta.gmst0_rad + OMEGA * t);
    const o = tab(data.moon_orient, t, 9);
    // Re-orthonormalise the interpolated rotation.
    const bx = V(o[0], o[3], o[6]).normalize(), bz = V(o[2], o[5], o[8]);
    bz.sub(bx.clone().multiplyScalar(bz.dot(bx))).normalize();
    const by = new THREE.Vector3().crossVectors(bz, bx);
    mBody.makeBasis(bx, by, bz).setPosition(moonC.clone().sub(s.p));
    moon.matrix.copy(mBody);
    moon.matrixWorldNeedsUpdate = true;
    const sd = V(...tab(data.sun, t, 3)).normalize();
    sun.position.copy(sd.multiplyScalar(1e7)); sun.target.position.set(0, 0, 0);

    // Camera
    const dMoon = s.m.length();
    let base, look = V(0, 3, 0).applyQuaternion(q);
    const upW = (dMoon < 6e7 ? s.m : s.p).clone().normalize();
    const lowOrbit = dMoon < 4.5e6;
    if (preset === "Chase" || (preset === "Towards the Moon" && lowOrbit)) {
      // Elevation measured from the local horizontal, so the nearby body sits below the spacecraft.
      const fwd = preset === "Chase" ? V(0, 1, 0).applyQuaternion(q) : velocity(t, true).normalize();
      let fh = fwd.clone().sub(upW.clone().multiplyScalar(fwd.dot(upW)));
      if (fh.lengthSq() < 1e-6) fh = new THREE.Vector3().crossVectors(upW, V(0, 0, 1));
      fh.normalize();
      const side = new THREE.Vector3().crossVectors(fh, upW).normalize();
      const d = (preset === "Chase" ? 45 : 70) * zoom;
      const a = (preset === "Chase" ? 2.4 : Math.PI - 0.25) + az, e = (preset === "Chase" ? 0.3 : 0.18) + el;
      base = fh.clone().multiplyScalar(Math.cos(e) * Math.cos(a)).add(side.multiplyScalar(Math.cos(e) * Math.sin(a)))
        .add(upW.clone().multiplyScalar(Math.sin(e))).multiplyScalar(d).add(look);
      if (preset !== "Chase") {
        // Look ahead along the track, down towards the horizon.
        const dip = Math.acos(Math.min(1, RM / dMoon));
        look = fh.clone().multiplyScalar(Math.cos(dip * 0.8) * 1e5).sub(upW.clone().multiplyScalar(Math.sin(dip * 0.8) * 1e5));
      }
      camera.up.copy(upW);
      camera.fov = preset === "Chase" ? 45 : 55;
    } else {
      const target = preset === "Towards the Moon" ? moonC.clone().sub(s.p) : s.p.clone().negate();
      const dir = target.clone().normalize();
      const R = preset === "Towards the Moon" ? RM : RE, D = target.length();
      camera.fov = Math.max(2, Math.min(60, 2.6 * Math.atan(R / D) * 180 / Math.PI + 2)) / Math.sqrt(zoom);
      const f = camera.fov * Math.PI / 180;
      const d = 16 / (f / 3);                                  // spacecraft spans about a third of the view
      const upV = V(0, 0, 1).sub(dir.clone().multiplyScalar(dir.z)).normalize();
      const side = new THREE.Vector3().crossVectors(dir, upV);
      const off = dir.clone().multiplyScalar(-d)
        .add(upV.clone().multiplyScalar(d * Math.tan(f * 0.22)))
        .add(side.multiplyScalar(d * Math.tan(f * 0.12)));
      base = off.applyAxisAngle(upV, az * 0.05).add(look);
      look = target.clone().add(look);
      camera.up.copy(upV);
    }
    camera.aspect = W / H; camera.updateProjectionMatrix();
    camera.position.copy(base);
    camera.lookAt(look);
    dirty = true;
  }

  let raf = null;
  const loop = () => {
    if (dirty) { renderer.render(scene, camera); dirty = false; }
    raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
  const setPreset = name => { preset = name; az = 0; el = 0; zoom = 1; update(T); };
  return {
    canvas: el$, update, setPreset, _debug: {scene, camera, moon, earth, stack}, presets: Object.keys(presets), info: () => info,
    dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); },
  };
}
