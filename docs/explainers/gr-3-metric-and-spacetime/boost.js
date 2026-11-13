// Spacetime diagram under a Lorentz boost. The grid of a moving observer is drawn over the
// rest frame's; hyperbolae of constant interval -t^2 + x^2 stay fixed, which is what makes
// the interval a geometric quantity. Drag the event to move it.
export function mountBoost(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const wrap = el("div", "gx-wrap"), cv = el("canvas", "gx-canvas"), side = el("div", "gx-side");
  const lab = el("label", "gx-lab"), span = el("span"), sl = el("input"); sl.type = "range"; Object.assign(sl, { min: -0.95, max: 0.95, step: 0.01, value: 0.5 });
  lab.append(span, sl); const out = el("div", "gx-out"); side.append(lab, out, el("div", "gx-dim", "Drag the black event. Blue: the rest frame's axes and grid. Orange: the axes and grid of an observer moving at speed v. Grey: curves of constant interval."));
  wrap.append(cv, side); host.append(wrap);
  const ctx = cv.getContext("2d"); let E = { t: 2.0, x: 1.0 }; const R = 4;
  function size() { const w = Math.min(cv.parentElement.clientWidth - 260, 520) || 420; const d = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = w + "px"; cv.width = w * d; cv.height = w * d; draw(); }
  const S = () => cv.width / (2 * R), X = (x) => cv.width / 2 + x * S(), Y = (t) => cv.height / 2 - t * S();
  function line(x0, t0, x1, t1, col, w, dash) { ctx.strokeStyle = col; ctx.lineWidth = w * (window.devicePixelRatio || 1); ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(X(x0), Y(t0)); ctx.lineTo(X(x1), Y(t1)); ctx.stroke(); ctx.setLineDash([]); }
  function draw() {
    const v = Number(sl.value), g = 1 / Math.sqrt(1 - v * v), d = window.devicePixelRatio || 1;
    span.textContent = "Observer's speed v = " + v.toFixed(2) + " c   (γ = " + g.toFixed(3) + ")";
    ctx.clearRect(0, 0, cv.width, cv.height);
    for (let k = -R; k <= R; k++) { line(k, -R, k, R, "#1f5fa833", 1); line(-R, k, R, k, "#1f5fa833", 1); }
    for (const s of [-1, 1]) for (const c of [1, 2, 3]) {           // hyperbolae -t^2 + x^2 = s c^2
      ctx.strokeStyle = "#88888899"; ctx.lineWidth = d; for (const br of [-1, 1]) { ctx.beginPath();
        for (let u = -2.5; u <= 2.5; u += 0.02) { const a = c * Math.cosh(u), b = c * Math.sinh(u);
          const [x, t] = s > 0 ? [br * a, b] : [b, br * a]; u === -2.5 ? ctx.moveTo(X(x), Y(t)) : ctx.lineTo(X(x), Y(t)); } ctx.stroke(); } }
    line(-R, -R, R, R, "#d9622b88", 1.2, [5, 4]); line(-R, R, R, -R, "#d9622b88", 1.2, [5, 4]);
    for (let k = -6; k <= 6; k++) {                               // moving frame grid: t' = k and x' = k lines
      const L = 8; line(g * (k + v * -L), g * (v * k - L), g * (k + v * L), g * (v * k + L), "#d9622b30", 1);
      line(g * (v * k - L), g * (k - v * L), g * (v * k + L), g * (k + v * L), "#d9622b30", 1); }
    line(0, -R, 0, R, "#1f5fa8", 2); line(-R, 0, R, 0, "#1f5fa8", 2);
    line(-R * v, -R, R * v, R, "#d9622b", 2); line(-R, -R * v, R, R * v, "#d9622b", 2);
    ctx.fillStyle = "#1f5fa8"; ctx.font = `${12 * d}px sans-serif`; ctx.fillText("ct", X(0.1), Y(R - 0.3)); ctx.fillText("x", X(R - 0.3), Y(0.15));
    ctx.fillStyle = "#d9622b"; ctx.fillText("ct′", X(R * v * 0.92 + 0.12), Y(R * 0.92)); ctx.fillText("x′", X(R * 0.92), Y(R * v * 0.92 + 0.15));
    ctx.fillStyle = "#000"; ctx.beginPath(); ctx.arc(X(E.x), Y(E.t), 6 * d, 0, 7); ctx.fill();
    const tp = g * (E.t - v * E.x), xp = g * (E.x - v * E.t), s2 = -E.t * E.t + E.x * E.x;
    out.innerHTML = `<div><b>Rest frame:</b> ct = ${E.t.toFixed(2)}, x = ${E.x.toFixed(2)}</div>` +
      `<div><b>Moving frame:</b> ct′ = ${tp.toFixed(2)}, x′ = ${xp.toFixed(2)}</div>` +
      `<div><b>Interval</b> −c²t² + x² = ${s2.toFixed(2)} in both (${s2 < -1e-9 ? "timelike" : s2 > 1e-9 ? "spacelike" : "null"})</div>`;
  }
  let drag = false; const pos = (e) => { const r = cv.getBoundingClientRect(), p = e.touches ? e.touches[0] : e;
    return { x: ((p.clientX - r.left) / r.width * 2 - 1) * R, t: (1 - (p.clientY - r.top) / r.height * 2) * R }; };
  cv.addEventListener("pointerdown", (e) => { drag = true; E = pos(e); draw(); cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => { if (drag) { E = pos(e); draw(); } });
  cv.addEventListener("pointerup", () => { drag = false; });
  sl.oninput = draw; new ResizeObserver(size).observe(wrap); size(); return host;
}
