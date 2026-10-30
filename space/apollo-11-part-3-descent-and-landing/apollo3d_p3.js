// CGI view of Eagle's descent for the Part 3 player. Built from primitives at real dimensions (metres).
// World frame: Moon-centred, mean equator and equinox of 16 July 1969, with a floating origin at the LM.
// The lunar surface near the landing site is a polar mesh centred on Tranquility Base, fine near the
// site (decimetres) and coarse at the horizon; the craters on it are illustrative, apart from the large
// named ones and West crater, which are placed at their approximate positions.

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
// Named craters along the approach: [name, lat, lon, diameter km] (approximate).
export const CRATERS = [
  ["Maskelyne", 2.2, 30.1, 23], ["Moltke", -0.6, 24.2, 6.5], ["Sabine", 1.4, 20.1, 30], ["Ritter", 2.0, 19.2, 29],
  ["Schmidt", 0.9, 18.8, 11], ["Arago", 6.2, 21.4, 26], ["Censorinus", -0.4, 32.7, 3.8], ["Secchi", 2.4, 43.5, 22],
  ["Taruntius", 5.5, 46.5, 56], ["Torricelli", -4.6, 28.5, 22], ["Theon Junior", -2.4, 15.8, 18],
];

export function createScene(THREE, {data, land, d3, width, height}) {
  const W = width, H = height;
  const renderer = new THREE.WebGLRenderer({antialias: true, logarithmicDepthBuffer: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(W, H);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);
  const camera = new THREE.PerspectiveCamera(50, W / H, 0.05, 2e10);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const RAD = Math.PI / 180;

  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.mapSize.set(4096, 4096);
  Object.assign(sun.shadow.camera, {left: -140, right: 140, top: 140, bottom: -140, near: 1, far: 3000});
  sun.shadow.bias = -0.0005;
  sun.shadow.camera.updateProjectionMatrix();
  scene.add(sun, sun.target);
  const ambient = new THREE.AmbientLight(0x8899aa, 0.12);
  scene.add(ambient);

  {
    const g = new THREE.BufferGeometry(), n = 3000, p = new Float32Array(n * 3);
    let seed = 13; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < n; i++) {
      const z = 2 * rnd() - 1, a = 2 * Math.PI * rnd(), r = Math.sqrt(1 - z * z);
      p.set([5e9 * r * Math.cos(a), 5e9 * r * Math.sin(a), 5e9 * z], 3 * i);
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial({color: 0xffffff, size: 1.1, sizeAttenuation: false})));
  }

  const meta = data.meta;
  const RS = meta.r_site * 1000;                   // landing-site radius, m
  const site = meta.landing_site;
  const unit = (lat, lon) => V(Math.cos(lat * RAD) * Math.cos(lon * RAD), Math.cos(lat * RAD) * Math.sin(lon * RAD), Math.sin(lat * RAD));
  const S = unit(site[0], site[1]);                 // body frame
  const E = V(-Math.sin(site[1] * RAD), Math.cos(site[1] * RAD), 0);
  const N = new THREE.Vector3().crossVectors(S, E);
  const appr = V(...meta.approach_body);
  const ah = appr.clone().sub(S.clone().multiplyScalar(appr.dot(S))).normalize();
  const aE = ah.dot(E), aN = ah.dot(N);            // downrange direction in local east/north

  // Height field (m) at local tangent coordinates (x east, y north, metres from the site).
  const craters = [];
  {
    let seed = 29; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    // West crater: about 190 m across, 30 m deep, 400 m uprange of the landing site; Little West, 60 m uprange.
    craters.push([-400 * aE, -400 * aN, 95, 28]);
    craters.push([-60 * aE + 8 * aN, -60 * aN - 8 * aE, 15, 4]);
    for (let i = 0; i < 500; i++) {
      const r = 40 + 4000 * Math.sqrt(rnd()), a = 2 * Math.PI * rnd();
      const R = 2 + 45 * Math.pow(rnd(), 3);
      craters.push([r * Math.cos(a), r * Math.sin(a), R, 0.22 * R]);
    }
    for (let i = 0; i < 260; i++) {
      const r = 3000 + 900000 * Math.pow(rnd(), 1.6), a = 2 * Math.PI * rnd();
      const R = 150 + 3500 * Math.pow(rnd(), 3);
      craters.push([r * Math.cos(a), r * Math.sin(a), R, 0.18 * R]);
    }
    for (const [, lat, lon, dkm] of CRATERS) {
      const u = unit(lat, lon), ang = Math.acos(Math.min(1, u.dot(S)));
      const t = u.clone().sub(S.clone().multiplyScalar(u.dot(S))).normalize();
      const d = ang * RS;
      craters.push([d * t.dot(E), d * t.dot(N), dkm * 500, Math.min(2500, dkm * 500 * 0.18)]);
    }
  }
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
  const mariaAt = (u) => {
    let m = 0;
    for (const [, lat, lon, r] of MARIA) {
      const c = unit(lat, lon), ang = Math.acos(Math.min(1, u.dot(c))) / RAD;
      if (ang < r) m = Math.max(m, Math.min(1, (r - ang) / 1.5));
    }
    return m;
  };

  // Polar surface mesh in the Moon's body frame, positions relative to the site point.
  const surface = new THREE.Group();
  {
    const nR = 230, nA = 360, r0 = 0.6, r1 = 950000;
    const q = Math.pow(r1 / r0, 1 / (nR - 1));
    const pos = new Float32Array((nR * nA + 1) * 3), col = new Float32Array((nR * nA + 1) * 3);
    const idx = [];
    const put = (k, x, y) => {
      const rho = Math.hypot(x, y);
      const ang = rho / RS;
      const t = rho > 0 ? E.clone().multiplyScalar(x / rho).add(N.clone().multiplyScalar(y / rho)) : E.clone();
      const u = S.clone().multiplyScalar(Math.cos(ang)).add(t.multiplyScalar(Math.sin(ang)));
      const h = heightAt(x, y);
      const p = u.clone().multiplyScalar(RS + h).sub(S.clone().multiplyScalar(RS));
      pos.set([p.x, p.y, p.z], 3 * k);
      const m = mariaAt(u);
      const n = 0.92 + 0.08 * Math.sin(x * 0.0007 + 1.3 * Math.cos(y * 0.0009)) * Math.cos(y * 0.0005);
      const b = (0.62 - 0.22 * m) * n;
      col.set([b, b * 0.985, b * 0.95], 3 * k);
    };
    put(0, 0, 0);
    for (let i = 0; i < nR; i++) {
      const r = r0 * Math.pow(q, i);
      for (let j = 0; j < nA; j++) {
        const a = 2 * Math.PI * j / nA;
        put(1 + i * nA + j, r * Math.cos(a), r * Math.sin(a));
      }
    }
    for (let j = 0; j < nA; j++) idx.push(0, 1 + j, 1 + (j + 1) % nA);
    for (let i = 0; i < nR - 1; i++) for (let j = 0; j < nA; j++) {
      const a = 1 + i * nA + j, b = 1 + i * nA + (j + 1) % nA, c = a + nA, d = b + nA;
      idx.push(a, c, b, b, c, d);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({vertexColors: true, roughness: 1, metalness: 0}));
    mesh.receiveShadow = true;
    surface.add(mesh);
    // Boulders around West crater's rim and scattered near the site.
    let seed = 41; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const nb = 420, rock = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 0),
      new THREE.MeshStandardMaterial({color: 0x8d8c88, roughness: 1, flatShading: true}), nb);
    rock.castShadow = true; rock.receiveShadow = true;
    const M4 = new THREE.Matrix4(), Q = new THREE.Quaternion();
    const [wx, wy] = craters[0];
    for (let i = 0; i < nb; i++) {
      let x, y;
      if (i < 320) { const a = 2 * Math.PI * rnd(), r = 70 + 110 * Math.sqrt(rnd()); x = wx + r * Math.cos(a); y = wy + r * Math.sin(a); }
      else { const a = 2 * Math.PI * rnd(), r = 15 + 300 * rnd(); x = r * Math.cos(a); y = r * Math.sin(a); }
      const s = 0.3 + (i < 320 ? 2.2 : 0.6) * Math.pow(rnd(), 2.5);
      const rho = Math.hypot(x, y), ang = rho / RS;
      const t = E.clone().multiplyScalar(x / rho).add(N.clone().multiplyScalar(y / rho));
      const u = S.clone().multiplyScalar(Math.cos(ang)).add(t.multiplyScalar(Math.sin(ang)));
      const p = u.clone().multiplyScalar(RS + heightAt(x, y) + 0.3 * s).sub(S.clone().multiplyScalar(RS));
      Q.setFromEuler(new THREE.Euler(rnd() * 6, rnd() * 6, rnd() * 6));
      M4.compose(p, Q, V(s * (0.8 + 0.4 * rnd()), s * 0.7, s));
      rock.setMatrixAt(i, M4);
    }
    surface.add(rock);
  }
  surface.matrixAutoUpdate = false;
  scene.add(surface);

  // Whole Moon, a little below the local mesh, for the view from orbit.
  const moonBall = new THREE.Mesh(new THREE.SphereGeometry(RS - 3000, 128, 64),
    new THREE.MeshStandardMaterial({color: 0x8f8e8a, roughness: 1}));
  moonBall.matrixAutoUpdate = false;
  scene.add(moonBall);

  // Earth, at its true distance and size.
  const earth = new THREE.Group();
  {
    const tw = 1024, th = 512, cv = document.createElement("canvas");
    cv.width = tw; cv.height = th;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#1d4e89"; ctx.fillRect(0, 0, tw, th);
    if (land) { const path = d3.geoPath(d3.geoEquirectangular().scale(tw / (2 * Math.PI)).translate([tw / 2, th / 2]), ctx); ctx.beginPath(); path(land); ctx.fillStyle = "#5b7a4a"; ctx.fill(); }
    const tex = new THREE.CanvasTexture(cv); tex.encoding = THREE.sRGBEncoding;
    const m = new THREE.Mesh(new THREE.SphereGeometry(6378e3, 48, 24), new THREE.MeshStandardMaterial({map: tex, roughness: 0.9}));
    m.rotation.x = Math.PI / 2;
    earth.add(m);
  }
  scene.add(earth);

  // ---- Eagle. Local axes: +y = LM +X (up, thrust axis), -z = LM +Z (forward, windows), +x = LM +Y (right). ----
  const gold = new THREE.MeshStandardMaterial({color: 0xc9a227, roughness: 0.35, metalness: 0.7});
  const black = new THREE.MeshStandardMaterial({color: 0x222222, roughness: 0.7});
  const grey = new THREE.MeshStandardMaterial({color: 0xa9adb3, roughness: 0.5, metalness: 0.3});
  const silver = new THREE.MeshStandardMaterial({color: 0xd8dade, roughness: 0.25, metalness: 0.85});
  const dark = new THREE.MeshStandardMaterial({color: 0x3a3d42, roughness: 0.6, metalness: 0.2});
  const lm = new THREE.Group();
  const LEG_H = 2.6;                                  // footpad to descent-stage base, m (approximate)
  {
    const box = (w, h, d, mat, x, y, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat); m.position.set(x, y, z); m.castShadow = true; lm.add(m); return m; };
    const ds = new THREE.Mesh(new THREE.CylinderGeometry(2.15, 2.15, 1.65, 8), gold);
    ds.rotation.y = Math.PI / 8; ds.position.y = LEG_H + 0.83; ds.castShadow = true; lm.add(ds);
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.75, 1.4, 20, 1, true),
      new THREE.MeshStandardMaterial({color: 0x2b2b2b, roughness: 0.4, metalness: 0.6, side: THREE.DoubleSide}));
    bell.position.y = LEG_H - 0.3; lm.add(bell);
    box(3.0, 2.3, 2.6, grey, 0, LEG_H + 1.65 + 1.15, -0.1);
    box(2.2, 1.0, 1.2, dark, 0, LEG_H + 1.65 + 2.6, 0.7);
    const port = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.5, 0.35, 16), silver);
    port.position.y = LEG_H + 1.65 + 2.45; lm.add(port);
    // Forward windows (two triangles) and hatch on the -z face.
    [-0.55, 0.55].forEach(x => {
      const w = new THREE.Mesh(new THREE.CircleGeometry(0.32, 3), black);
      w.position.set(x, LEG_H + 1.65 + 1.75, -1.42); w.rotation.y = Math.PI; lm.add(w);
    });
    box(0.85, 0.85, 0.05, dark, 0, LEG_H + 1.65 + 0.6, -1.42);
    // RCS quads.
    [[1.6, 1.4], [-1.6, 1.4], [1.6, -1.6], [-1.6, -1.6]].forEach(([x, z]) => box(0.35, 0.35, 0.35, grey, x, LEG_H + 1.65 + 1.9, z));
    // Legs: struts to footpads at the four diagonals; probes under three pads (not the ladder leg, +z... front).
    for (let k = 0; k < 4; k++) {
      const a = Math.PI / 4 + k * Math.PI / 2, cx = Math.cos(a), cz = Math.sin(a);
      const top = V(1.9 * cx, LEG_H + 1.2, 1.9 * cz), pad = V(4.3 * cx, 0.1, 4.3 * cz);
      const strut = top.clone().sub(pad), L = strut.length();
      const s = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, L, 8), gold);
      s.position.copy(top.clone().add(pad).multiplyScalar(0.5));
      s.quaternion.setFromUnitVectors(V(0, 1, 0), strut.normalize()); s.castShadow = true; lm.add(s);
      const fp = new THREE.Mesh(new THREE.CylinderGeometry(0.47, 0.47, 0.18, 16), silver);
      fp.position.copy(pad); fp.castShadow = true; lm.add(fp);
      const isFront = Math.abs(a - 1.5 * Math.PI) < 0.8 || (cz < 0 && Math.abs(cx) < 0.8);
      if (!(cz < -0.5 && Math.abs(cx) < 0.75)) {
        const probe = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.7, 6), silver);
        probe.position.set(pad.x, pad.y - 0.85, pad.z); lm.add(probe);
      }
    }
  }
  const plumeMat = new THREE.MeshBasicMaterial({color: 0xd8e4ff, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide});
  const plume = new THREE.Mesh(new THREE.ConeGeometry(1.5, 9, 24, 1, true), plumeMat);
  plume.position.y = LEG_H - 1.0 - 4.5;
  lm.add(plume);
  scene.add(lm);
  // Dust sheet blown radially from under the engine.
  const dustTex = (() => {
    const cv = document.createElement("canvas"); cv.width = cv.height = 256;
    const c = cv.getContext("2d"), g = c.createRadialGradient(128, 128, 10, 128, 128, 128);
    g.addColorStop(0, "rgba(220,214,200,0)"); g.addColorStop(0.25, "rgba(220,214,200,0.9)");
    g.addColorStop(0.6, "rgba(220,214,200,0.35)"); g.addColorStop(1, "rgba(220,214,200,0)");
    c.fillStyle = g; c.fillRect(0, 0, 256, 256);
    for (let k = 0; k < 90; k++) {             // radial streaks
      const a = Math.random() * 2 * Math.PI; c.strokeStyle = "rgba(240,235,225,0.10)"; c.lineWidth = 2;
      c.beginPath(); c.moveTo(128 + 30 * Math.cos(a), 128 + 30 * Math.sin(a)); c.lineTo(128 + 126 * Math.cos(a), 128 + 126 * Math.sin(a)); c.stroke();
    }
    return new THREE.CanvasTexture(cv);
  })();
  const dust = new THREE.Mesh(new THREE.PlaneGeometry(80, 80),
    new THREE.MeshBasicMaterial({map: dustTex, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide}));
  scene.add(dust);

  // Columbia (CSM) for the undocking.
  const csm = new THREE.Group();
  {
    const cyl = (r0, r1, h, mat, y) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r0, h, 32), mat); m.position.y = y + h / 2; csm.add(m); };
    cyl(1.96, 1.96, 3.9, silver, 0); cyl(1.96, 0.45, 3.2, silver, 3.9);
    const bell = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 1.25, 2.9, 20, 1, true), new THREE.MeshStandardMaterial({color: 0x2b2b2b, side: THREE.DoubleSide}));
    bell.position.y = -1.45; csm.add(bell);
  }
  scene.add(csm);

  // ---- Lookups ----
  const tr = data.track, times = tr.map(d => d[0]);
  const tabAt = (rows, t, n, keyCache) => {
    const ts = rows._t ?? (rows._t = rows.map(d => d[0]));
    const i = Math.max(0, Math.min(rows.length - 2, d3.bisectRight(ts, t) - 1));
    const a = rows[i], b = rows[i + 1], f = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0])));
    return d3.range(1, n + 1).map(j => a[j] + f * (b[j] - a[j]));
  };
  const lmPos = t => V(...tabAt(tr, t, 3)).multiplyScalar(1000);
  const lmVel = t => lmPos(t + 2).sub(lmPos(t - 2)).multiplyScalar(0.25);
  const bodyMatrix = t => {
    const o = tabAt(data.moon_orient, t, 9);
    const bx = V(o[0], o[3], o[6]).normalize(), bz = V(o[2], o[5], o[8]);
    bz.sub(bx.clone().multiplyScalar(bz.dot(bx))).normalize();
    const by = new THREE.Vector3().crossVectors(bz, bx);
    return new THREE.Matrix4().makeBasis(bx, by, bz);
  };
  const thrustDir = t => (t >= meta.pdi && t <= meta.engine_stop) ? V(...tabAt(data.thrust, t, 3)).normalize() : null;
  const throttleAt = t => tabAt(tr, t, 11)[10];
  const altFtAt = t => tabAt(tr, t, 4)[3];

  // Attitude: thrust axis and window direction.
  const attitude = (t, p) => {
    const up = p.clone().normalize();
    let X, Zw;
    if (t > meta.engine_stop + 1) {
      X = up.clone();
      const f = appr.clone().applyMatrix4(bodyMatrix(t));
      Zw = f.sub(up.clone().multiplyScalar(f.dot(up))).normalize();
    } else {
      const v = lmVel(t);
      X = thrustDir(t) ?? v.clone().normalize().negate();
      const fwd = v.clone().sub(up.clone().multiplyScalar(v.dot(up))).normalize();
      const zUp = up.clone().add(fwd.clone().multiplyScalar(0.02));
      zUp.sub(X.clone().multiplyScalar(zUp.dot(X))).normalize();
      // Windows down until the yaw manoeuvre, then up; the yaw turns the LM about its thrust axis.
      const f = Math.max(0, Math.min(1, (t - meta.yaw[0]) / (meta.yaw[1] - meta.yaw[0])));
      const ang = Math.PI * (1 - (0.5 - 0.5 * Math.cos(Math.PI * f)));
      Zw = zUp.clone().applyAxisAngle(X, ang);
    }
    const Y = new THREE.Vector3().crossVectors(Zw, X);
    return new THREE.Matrix4().makeBasis(Y, X, Zw.clone().negate());
  };

  // ---- Camera ----
  let preset = "Chase", az = 0, el = 0, zoom = 1;
  const presets = ["Chase", "Aldrin's window", "From the surface"];
  const el$ = renderer.domElement;
  el$.style.cssText = "display:block;width:100%;height:auto;border-radius:6px;touch-action:none;cursor:grab";
  let drag = null, T = meta.pdi, dirty = true, info = {label: ""};
  el$.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY]; el$.setPointerCapture(e.pointerId); });
  el$.addEventListener("pointermove", e => {
    if (!drag) return;
    az += (e.clientX - drag[0]) * 0.008; el = Math.max(-1.2, Math.min(1.2, el + (e.clientY - drag[1]) * 0.006));
    drag = [e.clientX, e.clientY]; update(T);
  });
  el$.addEventListener("pointerup", () => { drag = null; });
  el$.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.15, Math.min(40, zoom * Math.exp(e.deltaY * 0.001))); update(T); }, {passive: false});

  function update(t) {
    T = t;
    const p = lmPos(t);                              // LM, Moon-relative (m)
    const up = p.clone().normalize();
    const B = bodyMatrix(t);
    // Surface and Moon relative to the LM.
    const siteW = S.clone().applyMatrix4(B).multiplyScalar(RS);
    surface.matrix.copy(B).setPosition(siteW.clone().sub(p));
    surface.matrixWorldNeedsUpdate = true;
    moonBall.matrix.makeTranslation(-p.x, -p.y, -p.z);
    moonBall.matrixWorldNeedsUpdate = true;
    const em = V(...tabAt(data.moon, t, 3)).multiplyScalar(-1000);   // Earth from the Moon's centre
    earth.position.copy(em.sub(p));
    const sd = V(...tabAt(data.sun, t, 3)).normalize();
    sun.position.copy(sd.clone().multiplyScalar(800)); sun.target.position.set(0, 0, 0);

    const R = attitude(t, p);
    lm.matrixAutoUpdate = true;
    lm.quaternion.setFromRotationMatrix(R);
    lm.position.set(0, 0, 0);
    const alt = altFtAt(t) * 0.3048;
    // Put the footpads on the ground after landing (the track's reference point is at the pads).
    const thr = t >= meta.pdi && t <= meta.engine_stop ? throttleAt(t) : 0;
    plume.visible = thr > 0.5;
    const fl = 0.9 + 0.1 * Math.sin(t * 37) * Math.sin(t * 11);
    plume.scale.set(fl * Math.sqrt(thr / 60), 0.4 + 0.8 * thr / 60, fl * Math.sqrt(thr / 60));
    // Dust when the engine is close to the ground.
    const dustK = thr > 0 && alt < 40 ? (1 - alt / 40) * 0.6 : 0;
    dust.material.opacity = dustK;
    dust.visible = dustK > 0.01;
    if (dust.visible) {
      dust.position.copy(up.clone().multiplyScalar(-alt + 0.3));
      dust.quaternion.setFromUnitVectors(V(0, 0, 1), up);
      dust.scale.setScalar(0.6 + 1.2 * (1 - alt / 40));
    }
    // CSM near the LM while they are close.
    const c = V(...tabAt(data.csm, t, 3)).multiplyScalar(1000).sub(p);
    csm.visible = c.length() < 20000 && t < meta.doi[0];
    if (csm.visible) {
      csm.position.copy(c);
      csm.quaternion.setFromUnitVectors(V(0, 1, 0), lmVel(t).normalize().negate());
    }
    const prog = t < meta.pdi ? (t < meta.doi[0] ? "Separation and inspection" : t < meta.doi[1] + 5 ? "Descent orbit insertion" : "Coasting down to 50,000 ft")
      : t < meta.p64 ? (t < meta.yaw[0] ? "Braking phase (P63), windows down" : t < meta.yaw[1] ? "Yawing to windows up" : "Braking phase (P63), windows up")
      : t < meta.p66 ? "Approach phase (P64)" : t < meta.touchdown ? "Armstrong flying (P66)" : "Tranquility Base";
    info = {label: `${prog}${thr > 0.5 ? ` · descent engine ${thr.toFixed(0)}%` : ""}`};

    // Camera
    const vel = lmVel(t);
    let fh = vel.clone().sub(up.clone().multiplyScalar(vel.dot(up)));
    if (fh.lengthSq() < 1e-6 || t > meta.engine_stop) fh = new THREE.Vector3(0, 0, -1).applyMatrix4(R).sub(up.clone().multiplyScalar(new THREE.Vector3(0, 0, -1).applyMatrix4(R).dot(up)));
    fh.normalize();
    const side = new THREE.Vector3().crossVectors(fh, up).normalize();
    if (preset === "Chase") {
      const d = 32 * zoom, a = 2.5 + az, e = 0.28 + el;
      const look = up.clone().multiplyScalar(3.5);
      camera.position.copy(fh.clone().multiplyScalar(Math.cos(e) * Math.cos(a)).add(side.clone().multiplyScalar(Math.cos(e) * Math.sin(a)))
        .add(up.clone().multiplyScalar(Math.sin(e))).multiplyScalar(d).add(look));
      // keep above the ground
      const above = camera.position.dot(up) + alt;
      if (above < 2) camera.position.add(up.clone().multiplyScalar(2 - above));
      camera.up.copy(up); camera.fov = 50; camera.lookAt(look);
    } else if (preset === "Aldrin's window") {
      const eye = V(0.7, LEG_H + 1.65 + 1.8, -1.2).applyMatrix4(R);
      const dir = V(0.3 + az * 0.5, -0.95 - el * 0.5, -1).normalize().applyMatrix4(R);
      camera.position.copy(eye);
      camera.up.copy(V(0, 1, 0).applyMatrix4(R)); camera.fov = 62 / Math.sqrt(zoom);
      camera.lookAt(eye.clone().add(dir));
    } else {
      // A point on the surface 150 m to the south of the site, eye height 1.7 m.
      const local = E.clone().multiplyScalar(-40).add(N.clone().multiplyScalar(-150));
      const pt = S.clone().multiplyScalar(RS).add(local);
      const ptW = pt.clone().normalize().multiplyScalar(RS + heightAt(-40, -150) + 1.7).applyMatrix4(B).sub(p);
      camera.position.copy(ptW);
      camera.up.copy(ptW.clone().add(p).normalize());
      const dist = ptW.length();
      camera.fov = Math.max(0.5, Math.min(60, 2 * Math.atan(25 / dist) / RAD)) / Math.sqrt(zoom);
      camera.lookAt(0, 0, 0);
    }
    camera.aspect = W / H; camera.updateProjectionMatrix();
    ambient.intensity = preset === "From the surface" ? 0.18 : 0.12;
    dirty = true;
  }

  let raf = null;
  const loop = () => { if (dirty) { renderer.render(scene, camera); dirty = false; } raf = requestAnimationFrame(loop); };
  raf = requestAnimationFrame(loop);
  const setPreset = name => { preset = name; az = 0; el = 0; zoom = 1; update(T); };
  return {
    canvas: el$, update, setPreset, presets, info: () => info,
    dispose: () => { cancelAnimationFrame(raf); renderer.dispose(); },
  };
}
