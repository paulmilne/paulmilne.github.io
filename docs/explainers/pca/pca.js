// PCA by hand: rotate a direction through a 2D cloud and watch the variance
// of the projections. Its maximum is the top eigenvector of the covariance.

function randn(r) { let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function mulberry(a) { return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

function stats(P) {
  const n = P.length; let mx = 0, my = 0;
  for (const [x, y] of P) { mx += x; my += y; } mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of P) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
  sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
  const T = sxx + syy, D = sxx * syy - sxy * sxy, s = Math.sqrt(Math.max(T * T / 4 - D, 0));
  const l1 = T / 2 + s, l2 = T / 2 - s;
  const th = 0.5 * Math.atan2(2 * sxy, sxx - syy); // angle of the top eigenvector
  return { mx, my, sxx, syy, sxy, l1, l2, th };
}
const varAlong = (S, phi) => S.sxx * Math.cos(phi) ** 2 + 2 * S.sxy * Math.sin(phi) * Math.cos(phi) + S.syy * Math.sin(phi) ** 2;
const f = (x, n = 2) => (Math.abs(x) < 5e-10 ? 0 : x).toFixed(n).replace("-", "−");
const deg = (r) => ((((r * 180) / Math.PI) % 180) + 180) % 180;

export function mountPCA(host) {
  host.innerHTML = "";
  const DPR = window.devicePixelRatio || 1;
  let seed = 3, stretch = 2.5, tilt = 30, phi = (100 * Math.PI) / 180;
  let base = [], extra = [];

  function sample() {
    const r = mulberry(seed); const t = (tilt * Math.PI) / 180, c = Math.cos(t), s = Math.sin(t);
    base = Array.from({ length: 160 }, () => { const a = randn(r) * 1.1 * Math.sqrt(stretch), b = randn(r) * 1.1 / Math.sqrt(stretch); return [c * a - s * b, s * a + c * b]; });
  }

  const wrap = document.createElement("div"); wrap.className = "pc-grid";
  const cv = document.createElement("canvas"); cv.className = "pc-canvas";
  const side = document.createElement("div"); side.className = "pc-side";
  const curve = document.createElement("canvas"); curve.className = "pc-curve";
  const info = document.createElement("div"); info.className = "pc-info";
  const slider = (label, min, max, step, val, on) => {
    const l = document.createElement("label"); const i = document.createElement("input");
    Object.assign(i, { type: "range", min, max, step, value: val });
    const span = document.createElement("span"); span.textContent = label;
    i.addEventListener("input", () => on(+i.value)); l.append(span, i); return l;
  };
  const btn = (t, on) => { const b = document.createElement("button"); b.textContent = t; b.onclick = on; return b; };
  const row = document.createElement("div"); row.className = "pc-row";
  row.append(
    btn("Snap to PC1", () => { phi = stats(base.concat(extra)).th; draw(); }),
    btn("New sample", () => { seed++; sample(); draw(); }),
    btn("Remove added points", () => { extra = []; draw(); })
  );
  side.append(
    slider("Elongation", 1, 8, 0.1, stretch, (v) => { stretch = v; sample(); draw(); }),
    slider("Tilt of the cloud", -90, 90, 1, tilt, (v) => { tilt = v; sample(); draw(); }),
    row,
    Object.assign(document.createElement("div"), { className: "pc-note", textContent: "Drag anywhere on the plot to turn the line. Shift-click (or long-press) to add a point, such as an outlier." }),
    curve, info
  );
  wrap.append(cv, side); host.append(wrap);

  let W = 0; const R = 5;
  const toPx = (x, y) => [W / 2 + (x / R) * (W / 2), W / 2 - (y / R) * (W / 2)];
  const fromPx = (px, py) => [((px - W / 2) / (W / 2)) * R, (-(py - W / 2) / (W / 2)) * R];

  function resize() {
    W = Math.min(440, wrap.clientWidth > 640 ? wrap.clientWidth * 0.55 : wrap.clientWidth);
    cv.style.width = cv.style.height = W + "px"; cv.width = cv.height = W * DPR;
    const cw = Math.max(200, Math.min(380, side.clientWidth || 300));
    curve.style.width = cw + "px"; curve.style.height = "130px"; curve.width = cw * DPR; curve.height = 130 * DPR;
    draw();
  }

  function draw() {
    if (!W) return;
    const P = base.concat(extra), S = stats(P);
    const ctx = cv.getContext("2d"); ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, W);
    ctx.strokeStyle = "#e3e3e3"; ctx.lineWidth = 1;
    for (let k = -R; k <= R; k++) { let a = toPx(k, -R), b = toPx(k, R); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); a = toPx(-R, k); b = toPx(R, k); ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); }
    const u = [Math.cos(phi), Math.sin(phi)];
    // residuals and projections
    ctx.lineWidth = 0.7; ctx.strokeStyle = "rgba(217,98,43,.35)";
    for (const [x, y] of P) {
      const dx = x - S.mx, dy = y - S.my, t = dx * u[0] + dy * u[1];
      const q = [S.mx + t * u[0], S.my + t * u[1]];
      ctx.beginPath(); ctx.moveTo(...toPx(x, y)); ctx.lineTo(...toPx(...q)); ctx.stroke();
    }
    // PC axes (faint)
    const e1 = [Math.cos(S.th), Math.sin(S.th)], e2 = [-e1[1], e1[0]];
    ctx.setLineDash([3, 4]); ctx.strokeStyle = "#2a9d6f"; ctx.lineWidth = 1;
    for (const e of [e1, e2]) { ctx.beginPath(); ctx.moveTo(...toPx(S.mx - 8 * e[0], S.my - 8 * e[1])); ctx.lineTo(...toPx(S.mx + 8 * e[0], S.my + 8 * e[1])); ctx.stroke(); }
    ctx.setLineDash([]);
    // points
    for (const [x, y] of base) { ctx.beginPath(); ctx.fillStyle = "rgba(31,95,168,.6)"; ctx.arc(...toPx(x, y), 2.6, 0, 7); ctx.fill(); }
    for (const [x, y] of extra) { ctx.beginPath(); ctx.fillStyle = "#c0392b"; ctx.arc(...toPx(x, y), 4, 0, 7); ctx.fill(); }
    // projections on the line
    for (const [x, y] of P) {
      const t = (x - S.mx) * u[0] + (y - S.my) * u[1];
      ctx.beginPath(); ctx.fillStyle = "rgba(217,98,43,.75)"; ctx.arc(...toPx(S.mx + t * u[0], S.my + t * u[1]), 1.8, 0, 7); ctx.fill();
    }
    // the line
    ctx.strokeStyle = "#d9622b"; ctx.lineWidth = 2.2; ctx.beginPath();
    ctx.moveTo(...toPx(S.mx - 9 * u[0], S.my - 9 * u[1])); ctx.lineTo(...toPx(S.mx + 9 * u[0], S.my + 9 * u[1])); ctx.stroke();

    // variance vs angle
    const c2 = curve.getContext("2d"), cw = curve.width / DPR, ch = 130;
    c2.setTransform(DPR, 0, 0, DPR, 0, 0); c2.fillStyle = "#fff"; c2.fillRect(0, 0, cw, ch);
    const L = 30, B = 22, top = 10, vmax = S.l1 * 1.1;
    const X = (a) => L + (a / 180) * (cw - L - 24), Y = (v) => ch - B - (v / vmax) * (ch - B - top);
    c2.strokeStyle = "#ccc"; c2.beginPath(); c2.moveTo(L, ch - B); c2.lineTo(cw - 24, ch - B); c2.moveTo(L, top); c2.lineTo(L, ch - B); c2.stroke();
    c2.fillStyle = "#777"; c2.font = "10px system-ui, sans-serif";
    for (const a of [0, 45, 90, 135, 180]) c2.fillText(a + "°", X(a) - 8, ch - 7);
    c2.save(); c2.translate(10, ch / 2 + 22); c2.rotate(-Math.PI / 2); c2.fillText("variance", 0, 0); c2.restore();
    c2.strokeStyle = "#1f5fa8"; c2.lineWidth = 1.6; c2.beginPath();
    for (let a = 0; a <= 180; a += 1) { const v = varAlong(S, (a * Math.PI) / 180); a ? c2.lineTo(X(a), Y(v)) : c2.moveTo(X(a), Y(v)); } c2.stroke();
    c2.setLineDash([3, 3]); c2.strokeStyle = "#2a9d6f";
    for (const [v, lab] of [[S.l1, "λ₁"], [S.l2, "λ₂"]]) { c2.beginPath(); c2.moveTo(L, Y(v)); c2.lineTo(cw - 24, Y(v)); c2.stroke(); c2.fillStyle = "#2a9d6f"; c2.fillText(lab, cw - 20, Y(v) + 4); }
    c2.setLineDash([]);
    const a0 = deg(phi), v0 = varAlong(S, phi);
    c2.fillStyle = "#d9622b"; c2.beginPath(); c2.arc(X(a0), Y(v0), 4.5, 0, 7); c2.fill();

    const share = (100 * v0) / (S.l1 + S.l2);
    const off = Math.min(Math.abs(a0 - deg(S.th)), 180 - Math.abs(a0 - deg(S.th)));
    info.innerHTML = `
      <div>Covariance matrix: <span class="pc-m">[[${f(S.sxx)}, ${f(S.sxy)}], [${f(S.sxy)}, ${f(S.syy)}]]</span></div>
      <div style="color:#2a9d6f">Eigenvalues λ₁ = ${f(S.l1)}, λ₂ = ${f(S.l2)}; PC1 at ${f(deg(S.th), 0)}°, holding ${f((100 * S.l1) / (S.l1 + S.l2), 0)}% of the variance</div>
      <div style="color:#d9622b"><b>Your line, at ${f(a0, 0)}°: variance ${f(v0)}</b> (${f(share, 0)}% of the total)${off < 1.5 ? " — this is PC1" : ""}</div>
      <div class="pc-dim">Sum of squared residuals is smallest exactly where the variance is largest.</div>`;
  }

  let drag = false, lastAdd = 0;
  const pt = (ev) => { const r = cv.getBoundingClientRect(); const t = ev.touches ? ev.touches[0] : ev; return fromPx(t.clientX - r.left, t.clientY - r.top); };
  const aim = (ev) => { const S = stats(base.concat(extra)); const [x, y] = pt(ev); if (Math.hypot(x - S.mx, y - S.my) > 0.2) { phi = Math.atan2(y - S.my, x - S.mx); draw(); } };
  let pressTimer = null;
  cv.addEventListener("mousedown", (ev) => { if (ev.shiftKey) { extra.push(pt(ev)); draw(); return; } drag = true; aim(ev); });
  window.addEventListener("mousemove", (ev) => { if (drag) aim(ev); });
  window.addEventListener("mouseup", () => (drag = false));
  cv.addEventListener("touchstart", (ev) => { ev.preventDefault(); const p = pt(ev); drag = true; pressTimer = setTimeout(() => { drag = false; extra.push(p); draw(); }, 600); }, { passive: false });
  cv.addEventListener("touchmove", (ev) => { ev.preventDefault(); clearTimeout(pressTimer); if (drag) aim(ev); }, { passive: false });
  cv.addEventListener("touchend", () => { clearTimeout(pressTimer); drag = false; });

  sample();
  new ResizeObserver(resize).observe(wrap);
  resize();
  return host;
}
