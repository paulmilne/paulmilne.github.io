// Parallel transport round a circle of latitude on a sphere. In the orthonormal frame
// (e_theta, e_phi) the transported vector turns at rate -cos(theta0) per radian of longitude
// (from the Christoffel symbols of the sphere), so after one circuit it has turned by
// 2 pi cos(theta0) relative to the frame: 2 pi (1 - cos theta0) relative to where it started,
// which equals the enclosed area divided by a^2.
export function mountTransport(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const wrap = el("div", "gx-wrap"), cv = el("canvas", "gx-canvas"), side = el("div", "gx-side");
  const mk = (min, max, step, val) => { const lab = el("label", "gx-lab"), s = el("span"), i = el("input"); i.type = "range"; Object.assign(i, { min, max, step, value: val }); lab.append(s, i); side.append(lab); return [i, s]; };
  const [lat, latL] = mk(5, 85, 1, 45), [lon, lonL] = mk(0, 360, 1, 360);
  const out = el("div", "gx-out"); side.append(out, el("div", "gx-dim", "Purple: the vector carried round the circle without turning, by parallel transport. Grey (at the end point): the direction it started in relative to the local frame, pointing due south."));
  wrap.append(cv, side); host.append(wrap); const ctx = cv.getContext("2d");
  const tilt = 0.45, ct = Math.cos(tilt), st = Math.sin(tilt);
  function size() { const w = Math.min(cv.parentElement.clientWidth - 260, 480) || 400, d = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = w + "px"; cv.width = w * d; cv.height = w * d; draw(); }
  const proj = (p) => { const y = p[1] * ct - p[2] * st, z = p[1] * st + p[2] * ct; const s = cv.width * 0.38; return [cv.width / 2 + p[0] * s, cv.height / 2 - z * s, y]; };
  const sph = (t, p) => [Math.sin(t) * Math.cos(p), Math.sin(t) * Math.sin(p), Math.cos(t)];
  const eth = (t, p) => [Math.cos(t) * Math.cos(p), Math.cos(t) * Math.sin(p), -Math.sin(t)], eph = (t, p) => [-Math.sin(p), Math.cos(p), 0];
  function arrow(P, D, col, w) { const d = window.devicePixelRatio || 1, a = proj(P), b = proj(P.map((v, i) => v + D[i] * 0.22));
    if (a[2] > 0.15 && b[2] > 0.15) ctx.globalAlpha = 0.25; ctx.strokeStyle = ctx.fillStyle = col; ctx.lineWidth = w * d;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), h = 7 * d; ctx.beginPath(); ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - h * Math.cos(ang - 0.4), b[1] - h * Math.sin(ang - 0.4)); ctx.lineTo(b[0] - h * Math.cos(ang + 0.4), b[1] - h * Math.sin(ang + 0.4)); ctx.fill(); ctx.globalAlpha = 1; }
  function draw() {
    const th0 = (90 - Number(lat.value)) * Math.PI / 180, L = Number(lon.value) * Math.PI / 180, d = window.devicePixelRatio || 1;
    latL.textContent = "Latitude of the circle: " + lat.value + "°"; lonL.textContent = "Carried round: " + lon.value + "° of longitude";
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#dde6f2"; ctx.beginPath(); ctx.arc(cv.width / 2, cv.height / 2, cv.width * 0.38, 0, 7); ctx.fill();
    ctx.strokeStyle = "#aaa"; ctx.lineWidth = d; for (const la of [-60, -30, 0, 30, 60]) { ctx.beginPath(); let first = true;
      for (let p = 0; p <= 6.3; p += 0.05) { const q = proj(sph((90 - la) * Math.PI / 180, p)); if (q[2] > 0) { first = true; continue; } first ? ctx.moveTo(q[0], q[1]) : ctx.lineTo(q[0], q[1]); first = false; } ctx.stroke(); }
    ctx.fillStyle = "#2a9d6f22"; ctx.beginPath(); for (let p = 0; p <= 6.3; p += 0.04) { const q = proj(sph(th0, p)); p === 0 ? ctx.moveTo(q[0], q[1]) : ctx.lineTo(q[0], q[1]); } ctx.fill();
    ctx.strokeStyle = "#2a9d6f"; ctx.lineWidth = 2 * d; ctx.beginPath(); for (let p = 0; p <= L + 1e-9; p += 0.02) { const q = proj(sph(th0, p)); p === 0 ? ctx.moveTo(q[0], q[1]) : ctx.lineTo(q[0], q[1]); } ctx.stroke();
    const n = 12; for (let k = 0; k <= n; k++) { const p = L * k / n, c = Math.cos(th0);
      const a = Math.cos(c * p), b = -Math.sin(c * p); const P = sph(th0, p), et = eth(th0, p), ep = eph(th0, p);
      if (k === n) arrow(P, et, "#999", 2);
      arrow(P, et.map((v, i) => a * v + b * ep[i]), "#7b4fa3", 2.4); }
    const c = Math.cos(th0), turn = c * L, area = 2 * Math.PI * (1 - c);
    // angle from the starting vector (e_theta at phi = 0) to the transported vector, after a full circuit,
    // measured in the tangent plane at the starting point (frames coincide there), anticlockwise seen from outside
    const fa = Math.cos(c * 2 * Math.PI), fb = -Math.sin(c * 2 * Math.PI);
    let ang = Math.atan2(fb, fa); if (ang < 0) ang += 2 * Math.PI;
    out.innerHTML = `<div><b>Turned relative to the local frame (e<sub>θ</sub>, e<sub>φ</sub>):</b> ${(turn * 180 / Math.PI).toFixed(1)}°</div>` +
      (Number(lon.value) === 360
        ? `<div><b>After the full circuit, the vector has turned by</b> ${(ang * 180 / Math.PI).toFixed(1)}°</div>` +
          `<div><b>Area of the cap inside the circle ÷ a²:</b> ${area.toFixed(4)} rad = ${(area * 180 / Math.PI).toFixed(1)}°</div>`
        : "<div>Move the second slider to 360° to complete the circuit.</div>");
  }
  lat.oninput = draw; lon.oninput = draw; new ResizeObserver(size).observe(wrap); size(); return host;
}
