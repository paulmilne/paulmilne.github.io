// kf.js — small linear-algebra helpers and the simulations for the Kalman filter explainer.
// Pure functions, no dependencies.

// ---------- random numbers --------------------------------------------------------------------
export function rng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  let spare = null;
  const n = () => { if (spare !== null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0); const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
  return {u, n};
}

// ---------- matrices as arrays of rows ------------------------------------------------------------
export const T = A => A[0].map((_, j) => A.map(r => r[j]));
export const mul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
export const add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
export const sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
export const scale = (A, s) => A.map(r => r.map(v => v * s));
export const eye = n => Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => +(i === j)));
export const col = v => v.map(x => [x]);
export const flat = A => A.map(r => r[0]);
export function inv(A) {                                   // Gauss–Jordan, small matrices
  const n = A.length, M = A.map((r, i) => [...r, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]; const d = M[c][c];
    for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map(r => r.slice(n));
}
export function chol2(P) { const a = Math.sqrt(Math.max(P[0][0], 1e-300)), b = P[1][0] / a, c = Math.sqrt(Math.max(P[1][1] - b * b, 0)); return [[a, 0], [b, c]]; }
// points on the k-sigma ellipse of a 2x2 covariance centred at m
export function ellipse(m, P, k = 2, n = 60) { const L = chol2(P); return Array.from({length: n + 1}, (_, i) => { const t = 2 * Math.PI * i / n, x = Math.cos(t) * k, y = Math.sin(t) * k; return [m[0] + L[0][0] * x, m[1] + L[1][0] * x + L[1][1] * y]; }); }

// ---------- generic linear Kalman filter step ----------------------------------------------------
export function predict(x, P, F, Q) { return {x: mul(F, x), P: add(mul(mul(F, P), T(F)), Q)}; }
export function update(x, P, z, H, R) {
  const y = sub(z, mul(H, x)), S = add(mul(mul(H, P), T(H)), R), Si = inv(S), K = mul(mul(P, T(H)), Si);
  const IKH = sub(eye(P.length), mul(K, H));
  const Pn = add(mul(mul(IKH, P), T(IKH)), mul(mul(K, R), T(K)));          // Joseph form
  const nis = mul(mul(T(y), Si), y)[0][0];
  return {x: add(x, mul(K, y)), P: Pn, K, y, S, nis};
}

// ---------- 1. one-dimensional random walk ----------------------------------------------------------
export function oneD({seed = 3, N = 60, q = 0.05, r = 1, qF = null, rF = null}) {
  const R = rng(seed); let x = 0, xe = 0, Pe = 4; const out = [];
  const qf = qF ?? q, rf = rF ?? r;
  for (let k = 0; k < N; k++) {
    x += Math.sqrt(q) * R.n(); const z = x + Math.sqrt(r) * R.n();
    const xp = xe, Pp = Pe + qf, K = Pp / (Pp + rf);
    xe = xp + K * (z - xp); Pe = (1 - K) * Pp;
    out.push({k, x, z, xp, Pp, xe, Pe, K});
  }
  return out;
}

// ---------- 2. constant-velocity tracking in 2D, with RTS smoother ---------------------------------
// state [px, vx, py, vy]; a manoeuvring target; noisy position fixes.
export function track({seed = 5, N = 80, dt = 1, qTrue = 0.05, r = 4, qF = 0.05, rF = 4, turn = 1}) {
  const R = rng(seed), F = [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]];
  const G = q => { const a = dt ** 3 / 3, b = dt ** 2 / 2, c = dt; return [[a * q, b * q, 0, 0], [b * q, c * q, 0, 0], [0, 0, a * q, b * q], [0, 0, b * q, c * q]]; };
  const H = [[1, 0, 0, 0], [0, 0, 1, 0]], Rm = [[rF, 0], [0, rF]], Qf = G(qF);
  let s = [0, 1.2, 0, 0.12];
  let x = col([0, 0, 0, 0]), P = [[100, 0, 0, 0], [0, 10, 0, 0], [0, 0, 100, 0], [0, 0, 0, 10]];
  const truth = [], meas = [], filt = [], preds = [];
  for (let k = 0; k < N; k++) {
    // truth: constant velocity with random accelerations, plus a coordinated turn in the middle
    const w = turn * (k > 30 && k < 50 ? 0.08 : 0);
    const [vx, vy] = [s[1] * Math.cos(w) - s[3] * Math.sin(w), s[1] * Math.sin(w) + s[3] * Math.cos(w)];
    const ax = Math.sqrt(qTrue) * R.n(), ay = Math.sqrt(qTrue) * R.n();
    s = [s[0] + vx * dt + ax * dt * dt / 2, vx + ax * dt, s[2] + vy * dt + ay * dt * dt / 2, vy + ay * dt];
    const z = [s[0] + Math.sqrt(r) * R.n(), s[2] + Math.sqrt(r) * R.n()];
    const pr = predict(x, P, F, Qf); preds.push(pr);
    const up = update(pr.x, pr.P, col(z), H, Rm); x = up.x; P = up.P;
    truth.push([...s]); meas.push(z); filt.push({x: flat(x), P, nis: up.nis, K: up.K});
  }
  // Rauch–Tung–Striebel smoother
  const smooth = new Array(N); smooth[N - 1] = {x: filt[N - 1].x, P: filt[N - 1].P};
  for (let k = N - 2; k >= 0; k--) {
    const Pf = filt[k].P, Pp = preds[k + 1].P, C = mul(mul(Pf, T(F)), inv(Pp));
    const xs = add(col(filt[k].x), mul(C, sub(col(smooth[k + 1].x), preds[k + 1].x)));
    const Ps = add(Pf, mul(mul(C, sub(smooth[k + 1].P, Pp)), T(C)));
    smooth[k] = {x: flat(xs), P: Ps};
  }
  const err = arr => Math.sqrt(arr.reduce((s2, e, k) => s2 + (e[0] - truth[k][0]) ** 2 + (e[1] - truth[k][2]) ** 2, 0) / N);
  return {truth, meas, filt, smooth,
    rms: {meas: err(meas.map(m => [m[0], m[1]])), filt: err(filt.map(f => [f.x[0], f.x[2]])), smooth: err(smooth.map(f => [f.x[0], f.x[2]]))},
    nisMean: filt.slice(5).reduce((s2, f) => s2 + f.nis, 0) / (N - 5)};
}

// ---------- 3. steady-state gain of the scalar random walk ------------------------------------------
export function steadyGain(q, r) { const Pp = (q + Math.sqrt(q * q + 4 * q * r)) / 2; return {Pp, K: Pp / (Pp + r)}; }

// ---------- 4. nonlinear transform: polar -> Cartesian ---------------------------------------------
export function polar({range = 10, bearingDeg = 60, sr = 0.3, sbDeg = 15, n = 1500, seed = 4, alpha = 1, kappa = 1}) {
  const R = rng(seed), b0 = bearingDeg * Math.PI / 180, sb = sbDeg * Math.PI / 180;
  const mc = Array.from({length: n}, () => { const rr = range + sr * R.n(), bb = b0 + sb * R.n(); return [rr * Math.cos(bb), rr * Math.sin(bb)]; });
  const mean = [mc.reduce((s, p) => s + p[0], 0) / n, mc.reduce((s, p) => s + p[1], 0) / n];
  const cov = [[0, 0], [0, 0]]; mc.forEach(p => { const d = [p[0] - mean[0], p[1] - mean[1]]; cov[0][0] += d[0] * d[0] / n; cov[0][1] += d[0] * d[1] / n; cov[1][1] += d[1] * d[1] / n; }); cov[1][0] = cov[0][1];
  // linearised (EKF): mean at f(mu), covariance J Σ Jᵀ
  const Jm = [[Math.cos(b0), -range * Math.sin(b0)], [Math.sin(b0), range * Math.cos(b0)]], S = [[sr * sr, 0], [0, sb * sb]];
  const lin = {m: [range * Math.cos(b0), range * Math.sin(b0)], P: mul(mul(Jm, S), T(Jm))};
  // unscented transform with 2n+1 sigma points
  const L = 2, lam = alpha * alpha * (L + kappa) - L, c = Math.sqrt(L + lam);
  const sig = [[range, b0], [range + c * sr, b0], [range - c * sr, b0], [range, b0 + c * sb], [range, b0 - c * sb]];
  const wm = [lam / (L + lam), ...Array(4).fill(1 / (2 * (L + lam)))], wc = [wm[0] + (1 - alpha * alpha + 2), ...wm.slice(1)];
  const Y = sig.map(([rr, bb]) => [rr * Math.cos(bb), rr * Math.sin(bb)]);
  const um = [Y.reduce((s, y, i) => s + wm[i] * y[0], 0), Y.reduce((s, y, i) => s + wm[i] * y[1], 0)];
  const uP = [[0, 0], [0, 0]]; Y.forEach((y, i) => { const d = [y[0] - um[0], y[1] - um[1]]; uP[0][0] += wc[i] * d[0] * d[0]; uP[0][1] += wc[i] * d[0] * d[1]; uP[1][1] += wc[i] * d[1] * d[1]; }); uP[1][0] = uP[0][1];
  return {mc, mean, cov, lin, ut: {m: um, P: uP, points: Y}};
}
