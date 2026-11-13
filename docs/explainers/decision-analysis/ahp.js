// AHP pairwise-comparison panel: weights are the principal eigenvector of the
// comparison matrix, found by power iteration; consistency from lambda_max.

const RI = { 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45 };
const COLS = ["#1f5fa8", "#2a9d6f", "#d9622b", "#b08d3c", "#7b4fa3"];
const WORDS = { 1: "equally important as", 2: "slightly more important than", 3: "moderately more important than",
  4: "moderately to strongly more important than", 5: "strongly more important than", 6: "strongly to very strongly more important than",
  7: "very strongly more important than", 8: "very to extremely more important than", 9: "extremely more important than" };

// slider position s in -8..8  <->  Saaty value
const toVal = (s) => (s >= 0 ? s + 1 : 1 / (1 - s));
const toPos = (a) => (a >= 1 ? Math.round(a) - 1 : -(Math.round(1 / a) - 1));
const fmtA = (a) => (a >= 1 ? String(+a.toFixed(2)) : "1/" + +(1 / a).toFixed(2));

export function principal(A, tol = 1e-12) {
  const n = A.length; let w = Array(n).fill(1 / n), it = 0;
  for (; it < 500; it++) {
    const y = A.map((r) => r.reduce((s, a, j) => s + a * w[j], 0));
    const t = y.reduce((s, v) => s + v, 0); const z = y.map((v) => v / t);
    const d = Math.max(...z.map((v, i) => Math.abs(v - w[i]))); w = z;
    if (d < tol) break;
  }
  const Aw = A.map((r) => r.reduce((s, a, j) => s + a * w[j], 0));
  const lam = Aw.reduce((s, v, i) => s + v / w[i], 0) / n;
  return { w, lam, it: it + 1 };
}

export async function mountAHP(host) {
  host.innerHTML = "";
  const data = await (await fetch(new URL("scenario.json", document.baseURI))).json();
  const C = data.crit, n = C.length, Dn = data.Dn, opts = data.opts;
  const pairs = []; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  const start = Object.fromEntries(pairs.map(([i, j]) => [`${i},${j}`, data.J[`${i},${j}`]]));
  let J = { ...start };

  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const grid = el("div", "ahp-grid");
  const left = el("div", "ahp-left"), right = el("div", "ahp-right");
  grid.append(left, right); host.append(grid);

  const sliders = {};
  left.append(el("div", "ahp-h", "Your pairwise judgements"));
  for (const [i, j] of pairs) {
    const row = el("div", "ahp-pair");
    const a = el("span", "ahp-c", C[i]); a.style.color = COLS[i];
    const b = el("span", "ahp-c ahp-cr", C[j]); b.style.color = COLS[j];
    const s = document.createElement("input"); Object.assign(s, { type: "range", min: -8, max: 8, step: 1, value: -toPos(J[`${i},${j}`]) }); // drag towards the criterion that matters more
    const say = el("div", "ahp-say");
    s.addEventListener("input", () => { J[`${i},${j}`] = toVal(-s.value); draw(); });
    row.append(a, s, b, say); left.append(row); sliders[`${i},${j}`] = { s, say };
  }
  const btns = el("div", "ahp-btns");
  const reset = el("button", null, "Reset to the example");
  reset.onclick = () => { J = { ...start }; for (const k in sliders) sliders[k].s.value = -toPos(J[k]); draw(); };
  const fix = el("button", null, "Fix the worst judgement");
  fix.onclick = () => { const { worst, w } = state; const [i, j] = worst.pair; const target = w[i] / w[j];
    const sv = Math.max(-8, Math.min(8, target >= 1 ? Math.round(target) - 1 : -(Math.round(1 / target) - 1)));
    J[`${i},${j}`] = toVal(sv); sliders[`${i},${j}`].s.value = -sv; draw(); };
  btns.append(reset, fix); left.append(btns);

  const matBox = el("div", "ahp-mat"), wBox = el("div", "ahp-w"), cBox = el("div", "ahp-cons"), rBox = el("div", "ahp-rank");
  right.append(el("div", "ahp-h", "Comparison matrix A"), matBox, el("div", "ahp-h", "Weights: the principal eigenvector of A"), wBox, cBox,
    el("div", "ahp-h", "Ranking of the ten platforms"), rBox);
  let state = {};

  function draw() {
    const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : i < j ? J[`${i},${j}`] : 1 / J[`${j},${i}`])));
    const { w, lam, it } = principal(A);
    const CI = (lam - n) / (n - 1), CR = CI / RI[n];
    // most inconsistent judgement: largest |log(a_ij * w_j / w_i)|
    let worst = { e: -1 };
    for (const [i, j] of pairs) { const e = Math.abs(Math.log(A[i][j] * w[j] / w[i])); if (e > worst.e) worst = { e, pair: [i, j] }; }
    state = { w, worst };

    for (const [i, j] of pairs) {
      const a = J[`${i},${j}`], { say } = sliders[`${i},${j}`];
      say.textContent = a >= 1 ? `${C[i]} is ${WORDS[Math.round(a)]} ${C[j]} (${fmtA(a)})` : `${C[j]} is ${WORDS[Math.round(1 / a)]} ${C[i]} (${fmtA(1 / a)})`;
      say.classList.toggle("ahp-bad", worst.pair[0] === i && worst.pair[1] === j && CR > 0.02);
    }
    // matrix
    const t = document.createElement("table");
    const hr = document.createElement("tr"); hr.append(el("th")); C.forEach((c, k) => { const th = el("th", null, c.slice(0, 4)); th.style.color = COLS[k]; hr.append(th); }); t.append(hr);
    A.forEach((r, i) => { const tr = document.createElement("tr"); const th = el("th", null, C[i].slice(0, 4)); th.style.color = COLS[i]; tr.append(th);
      r.forEach((a, j) => { const td = el("td", null, fmtA(a)); const g = Math.log(a) / Math.log(9); td.style.background = g > 0 ? `rgba(217,98,43,${0.6 * g})` : `rgba(31,95,168,${-0.6 * g})`;
        if (worst.pair && ((worst.pair[0] === i && worst.pair[1] === j) || (worst.pair[0] === j && worst.pair[1] === i)) && CR > 0.02) td.style.outline = "2px solid #c0392b"; tr.append(td); });
      t.append(tr); });
    matBox.replaceChildren(t);
    // weights
    wBox.replaceChildren(...w.map((v, k) => { const r = el("div", "ahp-bar"); const lab = el("span", null, C[k]); const bar = el("span", "ahp-fill"); bar.style.width = `${(v * 100) / 0.6}%`; bar.style.background = COLS[k];
      const num = el("span", "ahp-num", (v * 100).toFixed(1) + "%"); r.append(lab, bar, num); return r; }));
    const [wi, wj] = worst.pair, implied = w[wi] / w[wj];
    cBox.innerHTML = `<div>λ<sub>max</sub> = ${lam.toFixed(3)} (n = ${n}), found by power iteration in ${it} steps</div>
      <div>Consistency index CI = (λ<sub>max</sub> − n)/(n − 1) = ${CI.toFixed(3)} · <b class="${CR < 0.1 ? "ahp-ok" : "ahp-no"}">CR = ${CR.toFixed(3)} ${CR < 0.1 ? "acceptable" : "too inconsistent (above 0.1)"}</b></div>
      ${CR > 0.02 ? `<div class="ahp-dim">Least consistent judgement: ${C[wi]} vs ${C[wj]}. You said ${fmtA(A[wi][wj])}; your other judgements imply about ${fmtA(implied)}.</div>` : `<div class="ahp-dim">Your judgements are almost perfectly consistent.</div>`}`;
    // ranking
    const vals = Dn.map((row) => row.reduce((s, x, k) => s + x * w[k], 0));
    const order = vals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    rBox.replaceChildren(...order.map(([v, i], r) => { const d = el("div", "ahp-bar"); const lab = el("span", null, `${r + 1}. ${opts[i]}`);
      const segs = el("span", "ahp-stack"); Dn[i].forEach((x, k) => { const s = el("span"); s.style.width = `${(x * w[k] * 100) / 0.9}%`; s.style.background = COLS[k]; s.title = `${C[k]}: ${(x * w[k]).toFixed(3)}`; segs.append(s); });
      d.append(lab, segs, el("span", "ahp-num", v.toFixed(3))); return d; }));
  }
  draw();
  return host;
}
