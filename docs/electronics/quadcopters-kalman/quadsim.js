// quadsim.js — small simulations for the quadcopter post (no dependencies).
// Exported functions are pure: given parameters and a seed, they return arrays for plotting.

export function rng(seed) {                    // mulberry32 + Box–Muller
  let a = seed >>> 0;
  const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  let spare = null;
  const n = () => { if (spare !== null) { const s = spare; spare = null; return s; } let x, y, r; do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0); const f = Math.sqrt(-2 * Math.log(r) / r); spare = y * f; return x * f; };
  return {u, n};
}

const D2R = Math.PI / 180;

// ---------- 1. one-axis attitude estimation ------------------------------------------------
// True roll angle follows a manoeuvre profile; gyro = rate + bias + noise; accelerometer-derived
// angle = angle + noise + error from linear acceleration. Four estimators run on the same data.
// Halfway through, the gyro bias jumps by biasStepDeg (a temperature change, say) so that the
// speed with which each estimator re-learns it is visible.
// The Kalman filter's angle process noise defaults to what the gyro's white noise actually
// contributes per step, gyroNoise * sqrt(dt), so that its model matches the simulated sensor.
export function attitudeRun(p) {
  const {seed = 1, T = 20, dt = 0.002, biasDeg = 3, biasStepDeg = 3, gyroNoiseDeg = 1.5, accNoiseDeg = 6, accelErr = 1,
         alpha = 0.995, qBiasDeg = 0.5, rDeg = 6} = p;
  const qAngleDeg = p.qAngleDeg ?? gyroNoiseDeg * Math.sqrt(dt);
  const R = rng(seed), n = Math.round(T / dt), every = 10;
  // manoeuvre: smoothed sequence of target angles
  const targets = [0, 15, -10, 25, 0, -20, 10, 0, 30, -5];
  let th = 0, w = 0, b = biasDeg * D2R;
  let gyroOnly = 0, comp = 0, kf = [0, 0], P = [[0.1, 0], [0, 0.05]];
  const out = {t: [], truth: [], gyro: [], acc: [], comp: [], kf: [], bias: [], biasEst: [], P00: []};
  let se = {gyro: 0, acc: 0, comp: 0, kf: 0}, cnt = 0;
  const q1 = (qAngleDeg * D2R) ** 2 * dt, q2 = (qBiasDeg * D2R) ** 2 * dt, r = (rDeg * D2R) ** 2;
  for (let k = 0; k < n; k++) {
    const t = k * dt, tgt = targets[Math.floor(t / 2) % targets.length] * D2R;
    // second-order attitude response to the target (what a controlled quad does)
    const wn = 6, zeta = 0.7, acc = wn * wn * (tgt - th) - 2 * zeta * wn * w;
    w += acc * dt; th += w * dt;
    b += R.n() * 0.02 * D2R * Math.sqrt(dt);                       // slow bias random walk
    if (k === Math.round(n / 2)) b += biasStepDeg * D2R;             // sudden bias change
    const gyro = w + b + R.n() * gyroNoiseDeg * D2R;
    const accAng = th + R.n() * accNoiseDeg * D2R + accelErr * 0.004 * acc; // linear accel corrupts tilt
    // estimators
    gyroOnly += gyro * dt;
    comp = alpha * (comp + gyro * dt) + (1 - alpha) * accAng;
    // Kalman: x = [angle, bias], u = gyro
    kf = [kf[0] + dt * (gyro - kf[1]), kf[1]];
    const P00 = P[0][0] - dt * (P[0][1] + P[1][0]) + dt * dt * P[1][1] + q1, P01 = P[0][1] - dt * P[1][1], P11 = P[1][1] + q2;
    const S = P00 + r, K0 = P00 / S, K1 = P01 / S, y = accAng - kf[0];
    kf = [kf[0] + K0 * y, kf[1] + K1 * y];
    P = [[(1 - K0) * P00, (1 - K0) * P01], [P01 - K1 * P00, P11 - K1 * P01]];
    if (t > T / 4) { se.gyro += (gyroOnly - th) ** 2; se.acc += (accAng - th) ** 2; se.comp += (comp - th) ** 2; se.kf += (kf[0] - th) ** 2; cnt++; }
    if (k % every === 0) {
      out.t.push(t); out.truth.push(th / D2R); out.gyro.push(gyroOnly / D2R); out.acc.push(accAng / D2R);
      out.comp.push(comp / D2R); out.kf.push(kf[0] / D2R); out.bias.push(b / D2R); out.biasEst.push(kf[1] / D2R); out.P00.push(Math.sqrt(P[0][0]) / D2R);
    }
  }
  out.rms = Object.fromEntries(Object.entries(se).map(([k, v]) => [k, Math.sqrt(v / cnt) / D2R]));
  return out;
}

// ---------- 2. planar quadcopter in closed loop ---------------------------------------------
// Side view: position (x, z), pitch angle th. Two thrusts at +/- l. Cascaded control:
// position PD -> desired tilt -> attitude P on the ESTIMATED angle -> rate P on the gyro
// (bias-corrected by the Kalman filter, which estimates it; true rate for "perfect knowledge").
// Position and velocity are taken as measured exactly, standing in for GNSS or a motion-capture system.
export function planarRun(p) {
  const {seed = 2, T = 40, dt = 0.002, est = "kf", biasDeg = 3, gyroNoiseDeg = 1.5, accNoiseDeg = 6,
         alpha = 0.995, qBiasDeg = 0.5, rDeg = 6} = p;
  const qAngleDeg = p.qAngleDeg ?? gyroNoiseDeg * Math.sqrt(dt);
  const R = rng(seed), n = Math.round(T / dt), m = 0.6, g = 9.81, l = 0.11, J = 0.004, drag = 0.25;
  const way = [[0, 2], [4, 2], [4, 4], [0, 4], [0, 2]];
  let x = 0, z = 2, vx = 0, vz = 0, th = 0, w = 0, b = biasDeg * D2R;
  let gyroInt = 0, comp = 0, kf = [0, 0], P = [[0.1, 0], [0, 0.05]], ax = 0, prevVx = 0;
  const q1 = (qAngleDeg * D2R) ** 2 * dt, q2 = (qBiasDeg * D2R) ** 2 * dt, r = (rDeg * D2R) ** 2;
  const out = {t: [], x: [], z: [], th: [], thEst: [], tx: [], tz: [], crashed: null};
  for (let k = 0; k < n; k++) {
    const t = k * dt, wi = Math.min(way.length - 1, Math.floor(t / (T / way.length)));
    const [tx, tz] = way[wi];
    // sensors
    const gyro = w + b + R.n() * gyroNoiseDeg * D2R;
    b += R.n() * 0.02 * D2R * Math.sqrt(dt);
    // accelerometer tilt estimate: with rotor drag the body-x specific force is about -drag*v; the naive
    // tilt formula atan2(f_x, f_z) therefore sees angle plus an acceleration-dependent error
    const accAng = th - ax / g * 0.5 + R.n() * accNoiseDeg * D2R;
    gyroInt += gyro * dt;
    comp = alpha * (comp + gyro * dt) + (1 - alpha) * accAng;
    kf = [kf[0] + dt * (gyro - kf[1]), kf[1]];
    const P00 = P[0][0] - dt * (P[0][1] + P[1][0]) + dt * dt * P[1][1] + q1, P01 = P[0][1] - dt * P[1][1], P11 = P[1][1] + q2;
    const S = P00 + r, K0 = P00 / S, K1 = P01 / S, yv = accAng - kf[0];
    kf = [kf[0] + K0 * yv, kf[1] + K1 * yv];
    P = [[(1 - K0) * P00, (1 - K0) * P01], [P01 - K1 * P00, P11 - K1 * P01]];
    const thEst = est === "gyro" ? gyroInt : est === "acc" ? accAng : est === "comp" ? comp : est === "truth" ? th : kf[0];
    // control
    const ex = tx - x, ez = tz - z;
    const thDes = Math.max(-0.5, Math.min(0.5, (0.9 * ex - 1.3 * vx) / g));     // pitch forward to move +x
    const Tz = m * (g + 3 * ez - 2.5 * vz) / Math.max(0.5, Math.cos(th));
    const wDes = 8 * (thDes - thEst);
    const rateEst = est === "kf" ? gyro - kf[1] : est === "truth" ? w : gyro;   // only the KF estimates the bias
    const tau = J * 60 * (wDes - rateEst);
    let T1 = Tz / 2 - tau / (2 * l), T2 = Tz / 2 + tau / (2 * l);
    T1 = Math.max(0, Math.min(8, T1)); T2 = Math.max(0, Math.min(8, T2));
    // dynamics
    const Ttot = T1 + T2, tq = (T2 - T1) * l;
    const axNew = (Ttot * Math.sin(th) - drag * vx) / m, az = (Ttot * Math.cos(th) - drag * vz) / m - g;
    ax = axNew; vx += ax * dt; vz += az * dt; x += vx * dt; z += vz * dt; w += tq / J * dt; th += w * dt;
    if (z < 0 || Math.abs(x) > 40 || Math.abs(th) > 1.4) { out.crashed = t; z = Math.max(z, 0); }
    if (k % 20 === 0) { out.t.push(t); out.x.push(x); out.z.push(z); out.th.push(th / D2R); out.thEst.push(thEst / D2R); out.tx.push(tx); out.tz.push(tz); }
    if (out.crashed !== null) break;
  }
  return out;
}

// ---------- 3. rotor momentum theory --------------------------------------------------------
export function hover({massG = 600, rotors = 4, diamIn = 5, FM = 0.55, etaElec = 0.8, CT = 0.1, battWh = 22, usable = 0.8, rho = 1.225}) {
  const g = 9.81, T = massG / 1000 * g / rotors, D = diamIn * 0.0254, A = Math.PI * D * D / 4;
  const vi = Math.sqrt(T / (2 * rho * A)), Pideal = T * vi, Pshaft = Pideal / FM, Pelec = Pshaft / etaElec;
  const nrev = Math.sqrt(T / (CT * rho * D ** 4));
  const Ptot = Pelec * rotors, minutes = battWh * usable / Ptot * 60;
  return {T, D, A, vi, Pideal, Pshaft, Pelec, rpm: nrev * 60, Ptot, minutes, diskLoading: T / A, gPerW: massG / Ptot};
}
