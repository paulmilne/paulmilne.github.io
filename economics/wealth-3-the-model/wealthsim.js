// Browser version of the top-1,000 model.
// Each year: every fortune grows by a random percentage (fat-tailed); the ten
// largest can be given an extra advantage; a fraction leave; new members arrive
// near the entry threshold; the top 1,000 are kept. Shares are relative, so a
// boom or slump that lifts everyone equally changes nothing.

function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gauss(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
// Student-t with 4 degrees of freedom, scaled to unit variance
function tshock(r) { const z = gauss(r); let c = 0; for (let i = 0; i < 4; i++) { const g = gauss(r); c += g * g; } return (z / Math.sqrt(c / 4)) / Math.SQRT2; }
const N = 1000;
const top10 = (x) => { let s = 0, t = 0; for (let i = 0; i < x.length; i++) { const w = Math.exp(x[i] - x[0]); t += w; if (i < 10) s += w; } return (100 * s) / t; };
const hill100 = (x) => { let s = 0; for (let i = 0; i < 99; i++) s += x[i] - x[99]; return 99 / s; };

export function step(x, P, r, entrants) {
  const y = [];
  for (let i = 0; i < x.length; i++) {
    if (r() < P.exit) continue;
    y.push(x[i] + P.sigma * tshock(r) + (i < 10 ? P.adv : 0));
  }
  y.sort((a, b) => b - a);
  const k = Math.round(P.turn), ref = y[Math.max(0, Math.min(N - k, y.length) - 1)];
  for (let i = 0; i < k; i++) y.push(ref + entrants[Math.floor(r() * entrants.length)]);
  y.sort((a, b) => b - a); y.length = Math.min(N, y.length);
  const m = y[y.length - 1]; for (let i = 0; i < y.length; i++) y[i] -= m;
  return y;
}

export async function mountWealthSim(host) {
  host.innerHTML = "";
  const D = await (await fetch(new URL("model.json", document.baseURI))).json();
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const P = { sigma: 0.25, turn: 155, exit: 0.055, adv: 0 };
  let r = rng(11), x, hist, year, timer = null;
  const PRE = { "2011–19": [0.25, 155, 0], "2020–22": [0.35, 205, 0], "2023–25": [0.275, 140, 0.105] };

  const grid = el("div", "ws-grid"), left = el("div"), side = el("div", "ws-side");
  const cv = el("canvas", "ws-canvas"), cv2 = el("canvas", "ws-canvas"); left.append(cv, cv2); grid.append(left, side); host.append(grid);
  const sliders = {};
  const slider = (key, label, min, max, stp, fmt) => {
    const l = el("label"), s = el("span"), i = el("input"); Object.assign(i, { type: "range", min, max, step: stp, value: P[key] });
    const show = () => (s.textContent = `${label}: ${fmt(P[key])}`); i.oninput = () => { P[key] = +i.value; show(); }; show();
    l.append(s, i); sliders[key] = { i, show }; return l;
  };
  const set = (v) => { [P.sigma, P.turn, P.adv] = v; for (const k of ["sigma", "turn", "adv"]) { sliders[k].i.value = P[k]; sliders[k].show(); } };
  const btn = (t, on, cls) => { const b = el("button", cls, t); b.onclick = on; return b; };
  const pre = el("div", "ws-row"); pre.append(el("span", "ws-dim", "Conditions of: "), ...Object.entries(PRE).map(([k, v]) => btn(k, () => set(v))));
  const run = btn("Run", () => (timer ? stop() : start()), "ws-go");
  const row = el("div", "ws-row"); row.append(run, btn("Step one year", () => { stop(); tick(); }),
    btn("Ten richest lose half", () => { for (let i = 0; i < 10; i++) x[i] += Math.log(0.5); x.sort((a, b) => b - a); hist.push({ y: year, v: top10(x), shock: true }); draw(); }));
  const row2 = el("div", "ws-row"); row2.append(btn("Restart from 2026", () => reset(2026)), btn("Restart from 2011", () => reset(2011)));
  const read = el("div", "ws-read");
  side.append(
    slider("sigma", "Spread of annual growth", 0.1, 0.5, 0.005, (v) => v.toFixed(3).replace(/0$/, "")),
    slider("turn", "New members a year", 0, 300, 5, (v) => `${v} of 1,000`),
    slider("adv", "Extra growth of the ten richest", -0.05, 0.2, 0.005, (v) => `${(v * 100).toFixed(1)}% a year`),
    pre, row, row2, read,
    el("div", "ws-dim", "The model is random, so each run differs. Black: actual, 2011–2026. Orange: this simulation."));

  function reset(y0) { stop(); r = rng(Math.floor(Math.random() * 1e9)); x = D[y0 === 2011 ? "start2011" : "start2026"].slice(); year = y0;
    hist = D.years.filter((y) => y <= y0).map((y, i) => ({ y, v: D.actual_t10[i], actual: true })); draw(); }
  function tick() { x = step(x, P, r, D.entrants); year++; hist.push({ y: year, v: top10(x) }); if (year >= 2100) stop(); draw(); }
  function start() { timer = setInterval(tick, 120); run.textContent = "Pause"; }
  function stop() { if (timer) clearInterval(timer); timer = null; run.textContent = "Run"; }

  const DPR = window.devicePixelRatio || 1; let Wd = 0;
  function size() { Wd = Math.min(left.clientWidth || 520, 620); for (const [c, h] of [[cv, 250], [cv2, 210]]) { c.style.width = Wd + "px"; c.style.height = h + "px"; c.width = Wd * DPR; c.height = h * DPR; } draw(); }
  function draw() {
    if (!Wd || !x) return;
    // top-10 share over time
    let c = cv.getContext("2d"), H = 250; c.setTransform(DPR, 0, 0, DPR, 0, 0); c.fillStyle = "#fff"; c.fillRect(0, 0, Wd, H);
    const L = 40, B = 24, T = 14, y0 = 2011, y1 = Math.max(2040, year + 2), vmax = Math.max(40, ...hist.map((h) => h.v)) * 1.08;
    const X = (y) => L + ((y - y0) / (y1 - y0)) * (Wd - L - 24), Y = (v) => H - B - (v / vmax) * (H - B - T);
    c.font = "11px system-ui, sans-serif"; c.fillStyle = "#777"; c.strokeStyle = "#e6e6e6";
    for (let v = 0; v <= vmax; v += vmax > 60 ? 20 : 10) { c.beginPath(); c.moveTo(L, Y(v)); c.lineTo(Wd - 10, Y(v)); c.stroke(); c.textAlign = "right"; c.fillText(v + "%", L - 5, Y(v) + 4); }
    c.textAlign = "center"; for (let y = 2015; y <= y1; y += y1 - y0 > 50 ? 10 : 5) c.fillText(y, X(y), H - 7);
    c.textAlign = "left"; c.fillText("share of the top 1,000's wealth held by the ten richest", L + 4, T);
    const seg = (pts, col, w) => { c.strokeStyle = col; c.lineWidth = w; c.beginPath(); pts.forEach((h, i) => (i ? c.lineTo(X(h.y), Y(h.v)) : c.moveTo(X(h.y), Y(h.v)))); c.stroke(); };
    const a = hist.filter((h) => h.actual), s = hist.filter((h) => !h.actual); seg(a, "#111", 2.2); if (s.length) seg([a[a.length - 1], ...s], "#d9622b", 2);
    // rank against wealth
    c = cv2.getContext("2d"); H = 210; c.setTransform(DPR, 0, 0, DPR, 0, 0); c.fillStyle = "#fff"; c.fillRect(0, 0, Wd, H);
    const xm = Math.max(6, x[0] * 1.05), RX = (v) => L + (v / xm) * (Wd - L - 10), RY = (k) => T + 6 + (Math.log(k) / Math.log(N)) * (H - B - T - 6);
    c.font = "11px system-ui, sans-serif"; c.fillStyle = "#777"; c.textAlign = "left"; c.fillText("rank (log scale) against wealth relative to the 1,000th fortune (log scale)", L + 4, T);
    c.textAlign = "right"; for (const k of [1, 10, 100, 1000]) c.fillText(k, L - 5, RY(k) + 4);
    c.textAlign = "center"; for (const m of [1, 10, 100, 1000, 10000]) if (Math.log(m) < xm) c.fillText("×" + m, RX(Math.log(m)), H - 7);
    c.fillStyle = "#1f5fa8"; for (let i = 0; i < x.length; i++) { c.beginPath(); c.arc(RX(x[i]), RY(i + 1), i < 10 ? 3 : 1.4, 0, 7); c.fill(); }
    read.innerHTML = `<b>${year}</b>: the ten richest hold <b>${top10(x).toFixed(1)}%</b>; the richest is ${Math.round(Math.exp(x[0])).toLocaleString()}× the 1,000th; tail exponent of the top 100: ${hill100(x).toFixed(2)}`;
  }
  new ResizeObserver(size).observe(host); reset(2026); size();
  return host;
}
