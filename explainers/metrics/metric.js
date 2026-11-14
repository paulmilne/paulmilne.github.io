// Metric explorer. Pick a metric written in some chart; the canvas shows the chart's coordinate
// plane, with the set {v : g(v, v) = eps^2} drawn at a grid of points: what a small circle of
// true radius eps looks like in these coordinates (a Tissot indicatrix). Click a point to read
// off the metric components there, and the true length of a unit step in each coordinate.
const M = {
  euclid: { name: "Plane, Cartesian coordinates (x, y)", lab: ["x", "y"], box: [-2, 2, -2, 2], eps: 0.18,
    g: (x, y) => [1, 0, 1], text: "g = dx² + dy². The same circle everywhere: the components are constant." },
  polar: { name: "Plane, polar coordinates (r, φ)", lab: ["r", "φ"], box: [0, 3, 0, 2 * Math.PI], eps: 0.14,
    g: (r, p) => [1, 0, r * r], text: "g = dr² + r² dφ². The same flat plane, in another chart: circles squash in φ as r grows, because one radian of φ covers a distance r." },
  latlong: { name: "Sphere, latitude–longitude (λ, φ)", lab: ["λ", "φ"], box: [-Math.PI, Math.PI, -1.45, 1.45], eps: 0.1,
    g: (l, p) => [Math.cos(p) ** 2, 0, 1], text: "g = cos²φ dλ² + dφ² (radius 1). Near the poles a degree of longitude is short, so a true circle covers many degrees east–west." },
  mercator: { name: "Sphere, Mercator (x, y)", lab: ["x", "y"], box: [-Math.PI, Math.PI, -2.6, 2.6], eps: 0.1,
    g: (x, y) => { const s = 1 / Math.cosh(y) ** 2; return [s, 0, s]; }, text: "g = sech²y (dx² + dy²). Conformal: circles stay circles, so angles are right, but sizes grow towards the poles. Greenland looks as big as Africa." },
  stereo: { name: "Sphere, stereographic (u, v)", lab: ["u", "v"], box: [-3, 3, -3, 3], eps: 0.12,
    g: (u, v) => { const s = 4 / (1 + u * u + v * v) ** 2; return [s, 0, s]; }, text: "g = 4(du² + dv²)/(1 + u² + v²)². Also conformal. The pole opposite the projection point sits at the centre; the projection point itself is at infinity." },
  poincare: { name: "Hyperbolic plane, Poincaré disc (x, y)", lab: ["x", "y"], box: [-1, 1, -1, 1], eps: 0.22, disc: true,
    g: (x, y) => { const s = 4 / (1 - x * x - y * y) ** 2; return [s, 0, s]; }, text: "g = 4(dx² + dy²)/(1 − x² − y²)². Curvature −1. The edge of the disc is infinitely far away: circles shrink without limit as they approach it." },
  shear: { name: "Plane, skewed coordinates (s, t)", lab: ["s", "t"], box: [-2, 2, -2, 2], eps: 0.18,
    g: (s, t) => [1, 0.6, 1], text: "g = ds² + 1.2 ds dt + dt². Flat, but the coordinate axes are not perpendicular (g₁₂ ≠ 0), so circles are tilted ellipses. Angle between the axes: cos θ = g₁₂/√(g₁₁g₂₂) = 0.6." },
};
export function mountMetric(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const wrap = el("div", "mx-wrap"), cv = el("canvas", "mx-canvas"), side = el("div", "mx-side");
  const sel = el("select", "mx-sel"); for (const k in M) { const o = el("option", null, M[k].name); o.value = k; sel.append(o); } sel.value = "polar";
  const desc = el("div", "mx-desc"), out = el("div", "mx-out");
  side.append(sel, desc, out, el("div", "mx-dim", "Each shape is a circle of the same true radius, drawn in the chart's coordinates. Click anywhere to inspect the metric there."));
  wrap.append(cv, side); host.append(wrap); const ctx = cv.getContext("2d"); let pick = null;
  function size() { const w = Math.min(cv.parentElement.clientWidth - 280, 460) || 380, d = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = w + "px"; cv.width = w * d; cv.height = w * d; draw(); }
  function draw() {
    const m = M[sel.value], [x0, x1, y0, y1] = m.box, d = window.devicePixelRatio || 1, W = cv.width, pad = 30 * d;
    const sc = Math.min((W - 2 * pad) / (x1 - x0), (W - 2 * pad) / (y1 - y0));
    const cx = (x) => W / 2 + (x - (x0 + x1) / 2) * sc, cy = (y) => W / 2 - (y - (y0 + y1) / 2) * sc;
    ctx.clearRect(0, 0, W, W); desc.textContent = m.text;
    ctx.strokeStyle = "#ccc"; ctx.lineWidth = d; ctx.strokeRect(cx(x0), cy(y1), (x1 - x0) * sc, (y1 - y0) * sc);
    if (m.disc) { ctx.strokeStyle = "#333"; ctx.beginPath(); ctx.arc(cx(0), cy(0), sc, 0, 7); ctx.stroke(); }
    ctx.fillStyle = "#555"; ctx.font = `${12 * d}px sans-serif`; ctx.fillText(m.lab[0] + " →", cx(x1) - 30 * d, cy(y0) + 18 * d); ctx.fillText(m.lab[1] + " ↑", cx(x0) - 26 * d, cy(y1) + 4 * d);
    const n = 9;
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
      const x = x0 + (x1 - x0) * (i + 0.5) / (n + 1), y = y0 + (y1 - y0) * (j + 0.5) / (n + 1);
      if (m.disc && x * x + y * y > 0.97) continue;
      ellipse(m, x, y, cx, cy, sc, "#1f5fa8");
    }
    if (pick) { ellipse(m, pick[0], pick[1], cx, cy, sc, "#d9622b"); ctx.fillStyle = "#d9622b"; ctx.beginPath(); ctx.arc(cx(pick[0]), cy(pick[1]), 3 * d, 0, 7); ctx.fill(); info(m, pick); }
    else out.innerHTML = "";
    cv._map = { x0, y0, x1, y1, sc, W };
  }
  function ellipse(m, x, y, cx, cy, sc, col) {
    const [a, b, c] = m.g(x, y); if (!isFinite(a) || a <= 0) return;
    // boundary of g(v,v) = eps^2: v = eps * R diag(1/sqrt(lambda)) (cos t, sin t)
    const tr = (a + c) / 2, det = a * c - b * b, disc = Math.sqrt(Math.max(0, tr * tr - det)), l1 = tr + disc, l2 = tr - disc;
    const th = Math.abs(b) < 1e-12 ? (a >= c ? 0 : Math.PI / 2) : Math.atan2(l1 - a, b);
    const r1 = m.eps / Math.sqrt(l1), r2 = m.eps / Math.sqrt(l2);
    if (r2 * sc > cv.width) return;
    ctx.save(); ctx.translate(cx(x), cy(y)); ctx.rotate(-th); ctx.beginPath(); ctx.ellipse(0, 0, r1 * sc, r2 * sc, 0, 0, 7);
    ctx.fillStyle = col + "44"; ctx.strokeStyle = col; ctx.lineWidth = window.devicePixelRatio || 1; ctx.fill(); ctx.stroke(); ctx.restore();
  }
  function info(m, [x, y]) {
    const [a, b, c] = m.g(x, y), f = (v) => Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(2) : v.toFixed(3);
    out.innerHTML = `<div><b>At (${m.lab[0]}, ${m.lab[1]}) = (${x.toFixed(2)}, ${y.toFixed(2)})</b></div>` +
      `<div class="mx-mat">g = [ ${f(a)}  ${f(b)} ; ${f(b)}  ${f(c)} ]</div>` +
      `<div>A small step d${m.lab[0]} has true length √g₁₁ · d${m.lab[0]} = ${f(Math.sqrt(a))} d${m.lab[0]}</div>` +
      `<div>A small step d${m.lab[1]} has true length √g₂₂ · d${m.lab[1]} = ${f(Math.sqrt(c))} d${m.lab[1]}</div>` +
      `<div>Area element √det g = ${f(Math.sqrt(a * c - b * b))} per unit of coordinate area</div>` +
      `<div>Angle between the coordinate axes: ${(Math.acos(b / Math.sqrt(a * c)) * 180 / Math.PI).toFixed(1)}°</div>`;
  }
  cv.addEventListener("pointerdown", (e) => { const r = cv.getBoundingClientRect(), mp = cv._map, d = window.devicePixelRatio || 1;
    const px = (e.clientX - r.left) * (cv.width / r.width), py = (e.clientY - r.top) * (cv.height / r.height);
    const x = (mp.x0 + mp.x1) / 2 + (px - mp.W / 2) / mp.sc, y = (mp.y0 + mp.y1) / 2 - (py - mp.W / 2) / mp.sc;
    const m = M[sel.value]; if (x < m.box[0] || x > m.box[1] || y < m.box[2] || y > m.box[3] || (m.disc && x * x + y * y >= 1)) return;
    pick = [x, y]; draw(); });
  sel.onchange = () => { pick = null; draw(); };
  new ResizeObserver(size).observe(wrap); size(); return host;
}
