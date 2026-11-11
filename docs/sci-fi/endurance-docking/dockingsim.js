// dockingsim.js: physics for the Endurance / Ranger docking (no rendering).
// World frame: origin at the Endurance's centre of mass, angular momentum along +Y.
// Endurance: torque-free rigid body (Euler's equations). Ranger: rigid body flown by an
// autopilot whose acceleration and angular acceleration are limited.

// ---------- small vector / quaternion / matrix helpers ----------
export const V = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  mul: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  norm: a => Math.hypot(a[0], a[1], a[2]),
  unit: a => { const n = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / n, a[1] / n, a[2] / n]; },
  clamp: (a, m) => { const n = Math.hypot(a[0], a[1], a[2]); return n > m ? [a[0] * m / n, a[1] * m / n, a[2] * m / n] : a; }
};
// quaternions [w, x, y, z]
export const Q = {
  mul: (a, b) => [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]],
  conj: a => [a[0], -a[1], -a[2], -a[3]],
  norm: a => { const n = Math.hypot(...a); return a.map(x => x / n); },
  rot: (q, v) => { const p = Q.mul(Q.mul(q, [0, ...v]), Q.conj(q)); return [p[1], p[2], p[3]]; },
  fromAxisAngle: (ax, ang) => { const u = V.unit(ax), s = Math.sin(ang / 2); return [Math.cos(ang / 2), u[0] * s, u[1] * s, u[2] * s]; },
  // integrate q with world-frame angular velocity w over dt
  stepWorld: (q, w, dt) => { const n = V.norm(w); if (n * dt < 1e-12) return q; return Q.norm(Q.mul(Q.fromAxisAngle(w, n * dt), q)); },
  // rotation vector (world) taking b to a: a = exp(v) b
  errVec: (a, b) => { let e = Q.mul(a, Q.conj(b)); if (e[0] < 0) e = e.map(x => -x);
    const s = Math.hypot(e[1], e[2], e[3]); if (s < 1e-12) return [0, 0, 0];
    const ang = 2 * Math.atan2(s, e[0]); return [e[1] / s * ang, e[2] / s * ang, e[3] / s * ang]; },
  fromMatrix: m => { // m: rows of a 3x3 rotation matrix
    const t = m[0][0] + m[1][1] + m[2][2]; let w, x, y, z;
    if (t > 0) { const s = Math.sqrt(t + 1) * 2; w = s / 4; x = (m[2][1] - m[1][2]) / s; y = (m[0][2] - m[2][0]) / s; z = (m[1][0] - m[0][1]) / s; }
    else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2; w = (m[2][1] - m[1][2]) / s; x = s / 4; y = (m[0][1] + m[1][0]) / s; z = (m[0][2] + m[2][0]) / s; }
    else if (m[1][1] > m[2][2]) { const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2; w = (m[0][2] - m[2][0]) / s; x = (m[0][1] + m[1][0]) / s; y = s / 4; z = (m[1][2] + m[2][1]) / s; }
    else { const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2; w = (m[1][0] - m[0][1]) / s; x = (m[0][2] + m[2][0]) / s; y = (m[1][2] + m[2][1]) / s; z = s / 4; }
    return Q.norm([w, x, y, z]);
  },
  between: (a, b) => { // shortest rotation taking unit a to unit b
    const c = V.cross(a, b), d = V.dot(a, b);
    if (d < -0.999999) { const ax = Math.abs(a[0]) < 0.9 ? V.cross(a, [1, 0, 0]) : V.cross(a, [0, 1, 0]); return Q.fromAxisAngle(ax, Math.PI); }
    return Q.norm([1 + d, c[0], c[1], c[2]]);
  }
};
// eigen-decomposition of a symmetric 3x3 (Jacobi); returns {values, vectors (columns)}
function eigSym(A) {
  const a = A.map(r => r.slice()), v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let it = 0; it < 50; it++) {
    let p = 0, q = 1;
    if (Math.abs(a[0][2]) > Math.abs(a[p][q])) { p = 0; q = 2; }
    if (Math.abs(a[1][2]) > Math.abs(a[p][q])) { p = 1; q = 2; }
    if (Math.abs(a[p][q]) < 1e-9 * (Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2]))) break;
    const th = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]), c = Math.cos(th), s = Math.sin(th);
    for (let k = 0; k < 3; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
    for (let k = 0; k < 3; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
    for (let k = 0; k < 3; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
  }
  const idx = [0, 1, 2].sort((i, j) => a[i][i] - a[j][j]);
  return { values: idx.map(i => a[i][i]), vectors: [0, 1, 2].map(r => idx.map(i => v[r][i])) };
}

// ---------- the Endurance ----------
export const RING_R = 29, BOX = [10, 8, 8], HUB_M = 40e3, HUB_PORT = 3.0;   // port face at z = -HUB_PORT (ring frame)
export const MASS = { engine: 50e3, cargo: 40e3, habitat: 30e3, cryo: 30e3, command: 30e3 };
export const LAYOUT = ["engine", "cargo", "habitat", "engine", "cargo", "habitat", "engine", "cargo", "cryo", "engine", "cargo", "command"];
export const LOST = [0, 1];
export function massProps(damaged) {
  const mods = LAYOUT.map((t, k) => ({ type: t, k, ang: k * Math.PI / 6, m: MASS[t], lost: damaged && LOST.includes(k) }));
  const live = mods.filter(m => !m.lost);
  let M = HUB_M, c = [0, 0, 0];
  for (const m of live) { const p = [RING_R * Math.cos(m.ang), RING_R * Math.sin(m.ang), 0]; M += m.m; c = V.add(c, V.mul(p, m.m)); }
  const com = V.mul(c, 1 / M);
  const I = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const addPoint = (m, p) => { const r = V.sub(p, com), r2 = V.dot(r, r);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) I[i][j] += m * ((i === j ? r2 : 0) - r[i] * r[j]); };
  addPoint(HUB_M, [0, 0, 0]); I[0][0] += HUB_M * 4; I[1][1] += HUB_M * 4; I[2][2] += HUB_M * 8;
  const [a, b, cc] = BOX;
  for (const m of live) {
    addPoint(m.m, [RING_R * Math.cos(m.ang), RING_R * Math.sin(m.ang), 0]);
    const It = m.m / 12 * (b * b + cc * cc), Ir = m.m / 12 * (a * a + cc * cc), Iz = m.m / 12 * (a * a + b * b);
    const t = [-Math.sin(m.ang), Math.cos(m.ang)], r = [Math.cos(m.ang), Math.sin(m.ang)];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) I[i][j] += It * t[i] * t[j] + Ir * r[i] * r[j];
    I[2][2] += Iz;
  }
  const e = eigSym(I);
  // make the principal frame right-handed with its third axis along +z of the ring
  let P = e.vectors; if (P[2][2] < 0) P = P.map(r => [r[0], -r[1], -r[2]]);
  const det = P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1]) - P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) + P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]);
  if (det < 0) P = P.map(r => [-r[0], r[1], r[2]]);
  // qP: rotation from principal frame to ring frame (columns of P are principal axes in ring coords)
  const qP = Q.fromMatrix(P);
  return { mods, M, com, I, Ip: e.values, qP };
}

// ---------- the Ranger (body: x forward, y up, z right) ----------
export const RANGER = { m: 40e3, port: [0, 1.6, 0], seat: [1.0, 0.2, 0] };
// docked pose in the ring frame: Ranger +y along ring +z, port face on the hub port
const qDock = Q.fromMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]);   // columns: Ranger x,y,z in ring coords = (1,0,0),(0,0,1),(0,-1,0)

export function createSim(opt) {
  const o = Object.assign({ rpm: 20, wobble: 3, damaged: true, aMax: 2.5 * 9.81, alphaMax: 0.6, standoff: 40 }, opt);
  const mp = massProps(o.damaged);
  const { Ip, qP, com } = mp;
  const w0 = o.rpm * 2 * Math.PI / 60, th = o.wobble * Math.PI / 180;
  // initial angular velocity in the ring frame, tilted from the ring's axis; convert to principal frame
  const wRing = [w0 * Math.sin(th), 0, w0 * Math.cos(th)];
  let wb = Q.rot(Q.conj(qP), wRing);
  const Lb = [Ip[0] * wb[0], Ip[1] * wb[1], Ip[2] * wb[2]];
  // orientation: principal frame -> world, chosen so that L points along +Y
  let qE = Q.between(V.unit(Lb), [0, 1, 0]);
  const s = {
    t: 0, mp, qE, wb, o,
    // ranger
    p: [0, -60, 6], v: [0, 0, 0], qR: Q.fromAxisAngle([0, 1, 0], 0.4), wR: [0, 0, 0],
    phase: "station", standoff: o.standoff, hold: 0, latched: false,
    aNeed: 0, aUsed: 0, saturated: false, posErr: 0, angErr: 0, seatG: 0, msg: ""
  };
  const ringToWorld = (q, x) => Q.rot(q, Q.rot(Q.conj(qP), V.sub(x, com)));   // ring coords -> world, given qE
  const qRingWorld = q => Q.mul(q, Q.conj(qP));
  s.ringToWorld = x => ringToWorld(s.qE, x);
  s.qRing = () => qRingWorld(s.qE);
  const eulerRHS = w => [(Ip[1] - Ip[2]) * w[1] * w[2] / Ip[0], (Ip[2] - Ip[0]) * w[2] * w[0] / Ip[1], (Ip[0] - Ip[1]) * w[0] * w[1] / Ip[2]];
  s.wWorld = () => Q.rot(s.qE, s.wb);
  s.wdotWorld = () => Q.rot(s.qE, eulerRHS(s.wb));                  // body-frame derivative = inertial derivative for omega
  // target pose of the Ranger for a given standoff
  function target(sOff) {
    // rigid part: where the Ranger would be if it were already docked; then back off along the
    // fixed angular-momentum axis (-Y), so the hold point does not swing with the wobble
    const qd = Q.mul(qRingWorld(s.qE), qDock);
    const pPortW = ringToWorld(s.qE, [0, 0, -HUB_PORT]);
    const rRig = V.sub(pPortW, Q.rot(qd, RANGER.port));
    const w = s.wWorld(), wd = s.wdotWorld();
    const sdot = s.phase === "approach" ? -1.0 : 0;
    const pd = V.add(rRig, [0, -sOff, 0]);
    const vd = V.add(V.cross(w, rRig), [0, sdot === 0 ? 0 : 1.0, 0]);
    const ad = V.add(V.cross(wd, rRig), V.cross(w, V.cross(w, rRig)));
    return { qd, pd, vd, ad, w, wd };
  }
  s.target = () => target(s.standoff);
  s.step = function (dt) {
    // Endurance: RK4 on Euler's equations, then rotate
    const f = eulerRHS, w = s.wb;
    const k1 = f(w), k2 = f(V.add(w, V.mul(k1, dt / 2))), k3 = f(V.add(w, V.mul(k2, dt / 2))), k4 = f(V.add(w, V.mul(k3, dt)));
    const wn = V.add(w, V.mul(V.add(V.add(k1, V.mul(k2, 2)), V.add(V.mul(k3, 2), k4)), dt / 6));
    const wmid = Q.rot(s.qE, V.mul(V.add(w, wn), 0.5));
    s.qE = Q.stepWorld(s.qE, wmid, dt); s.wb = wn; s.t += dt;
    // Ranger
    if (s.phase === "approach") s.standoff = Math.max(0, s.standoff - 1.0 * dt);
    const T = target(s.standoff);
    s.alphaNeed = V.norm(T.wd);
    if (s.latched) { s.p = T.pd; s.v = T.vd; s.qR = T.qd; s.wR = T.w; s.aNeed = V.norm(T.ad); s.aUsed = s.aNeed; s.saturated = false; s.posErr = 0; s.angErr = 0; }
    else {
      const ep = V.sub(T.pd, s.p), ev = V.sub(T.vd, s.v);
      const aCmd = V.add(T.ad, V.add(V.mul(ep, 1.2), V.mul(ev, 2.2)));
      const a = V.clamp(aCmd, o.aMax);
      s.aNeed = V.norm(T.ad); s.aUsed = V.norm(a); s.saturated = V.norm(aCmd) > o.aMax * 0.999;
      s.v = V.add(s.v, V.mul(a, dt)); s.p = V.add(s.p, V.mul(s.v, dt));
      const er = Q.errVec(T.qd, s.qR), ew = V.sub(T.w, s.wR);
      const al = V.clamp(V.add(T.wd, V.add(V.mul(er, 1.5), V.mul(ew, 2.4))), o.alphaMax);
      s.wR = V.add(s.wR, V.mul(al, dt)); s.qR = Q.stepWorld(s.qR, s.wR, dt);
      s.posErr = V.norm(ep); s.angErr = V.norm(er) * 180 / Math.PI;
      s.relSpeed = V.norm(ev);
      // seat load: acceleration of the seat point
      const rs = Q.rot(s.qR, RANGER.seat);
      const aSeat = V.add(a, V.add(V.cross(al, rs), V.cross(s.wR, V.cross(s.wR, rs))));
      s.seatG = V.norm(aSeat) / 9.81;
      // phase logic
      const good = s.posErr < 0.5 && s.angErr < 2 && s.relSpeed < 0.5;
      if (s.phase === "station") { s.hold = good ? s.hold + dt : 0; if (s.hold > 2) s.phase = "approach"; }
      if (s.phase === "approach" && s.standoff <= 0) {
        if (s.posErr < 0.3 && s.angErr < 2 && s.relSpeed < 0.3) { s.latched = true; s.phase = "latched"; }
        else s.phase = "missed";
      }
    }
    if (s.latched) { const rs = Q.rot(s.qR, RANGER.seat); const ws = s.wR; s.seatG = V.norm(V.add(V.mul(T.ad, 1), V.cross(ws, V.cross(ws, rs)))) / 9.81; }
    s.relSpeed = s.relSpeed || 0;
  };
  // diagnostics
  s.endRpm = () => V.norm(s.wWorld()) * 60 / (2 * Math.PI);
  s.rangerRpm = () => V.norm(s.wR) * 60 / (2 * Math.PI);
  s.axisTilt = () => { const n = Q.rot(s.qRing(), [0, 0, 1]); return Math.acos(Math.min(1, n[1])) * 180 / Math.PI; };
  s.portSpeed = () => { const p = s.ringToWorld([0, 0, -HUB_PORT]); return V.norm(V.cross(s.wWorld(), p)); };
  return s;
}
