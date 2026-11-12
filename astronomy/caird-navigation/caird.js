// caird.js: Monte Carlo of the James Caird voyage in the browser (same model as the post's figures).
export const GEOM = {"SG": [[534.79, 426.0], [545.92, 424.8], [558.64, 421.2], [566.59, 420.0], [577.4, 416.4], [585.67, 411.0], [593.62, 402.0], [599.97, 393.0], [606.97, 378.0], [598.38, 373.2], [588.85, 381.0], [579.31, 390.0], [569.77, 399.0], [563.41, 408.0], [558.0, 412.8], [549.1, 417.0], [539.56, 421.2]], "PW": [0.0, 0.0], "SWC": [[534.79, 426.0], [539.56, 421.2], [549.1, 417.0], [558.0, 412.8], [563.41, 408.0], [569.77, 399.0], [579.31, 390.0], [588.85, 381.0], [598.38, 373.2], [606.97, 378.0]], "KHB": [558.64, 414.0]};
export function mountCaird(host, G = GEOM) {
  host.innerHTML = `<div class="cd-grid"><canvas class="cd-map"></canvas><div class="cd-side">
    <label>Aim point (nmi down the coast) <input type="range" data-k="aim" min="0" max="60" step="5" value="30"><output></output></label>
    <label>Error of each sun sight (nmi) <input type="range" data-k="sig" min="2" max="30" step="1" value="5"><output></output></label>
    <label>Steering error (degrees) <input type="range" data-k="hdg" min="0" max="15" step="1" value="6"><output></output></label>
    <label>Unknown current (nmi/day) <input type="range" data-k="cur" min="0" max="10" step="1" value="4"><output></output></label>
    <div class="cd-sights">Sights: ${["26 Apr", "3 May", "4 May", "7 May"].map((d, i) => `<label><input type="checkbox" data-s="${i}" checked> ${d}</label>`).join("")}</div>
    <button class="cd-run">Run 300 voyages</button><div class="cd-out"></div></div></div>`;
  const cv = host.querySelector("canvas"), ctx = cv.getContext("2d"), $ = q => host.querySelector(q);
  const SD = [2, 9, 10, 13];
  const inPoly = (x, y, P) => { let c = false; for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
    const [xi, yi] = P[i], [xj, yj] = P[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
  const coastPoint = d => { const S = G.SWC; let acc = 0; for (let i = 0; i < S.length - 1; i++) { const dx = S[i + 1][0] - S[i][0], dy = S[i + 1][1] - S[i][1], L = Math.hypot(dx, dy);
      if (acc + L >= d || i === S.length - 2) { const f = Math.min(1, (d - acc) / L); return [S[i][0] + f * dx, S[i][1] + f * dy]; } acc += L; } };
  const randn = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); };
  const NW = G.SG[0], SE = G.SG[8];
  function run() {
    const p = {}; host.querySelectorAll("[data-k]").forEach(i => p[i.dataset.k] = +i.value);
    const sights = SD.filter((d, i) => host.querySelector(`[data-s="${i}"]`).checked);
    const tgt = coastPoint(p.aim), n = 300, dt = 0.25, tracks = [], res = [0, 0, 0, 0];
    for (let k = 0; k < n; k++) {
      let tx = G.PW[0], ty = G.PW[1], ex = tx, ey = ty, out = 0; const tr = [[tx, ty]];
      const bias = randn() * p.hdg, cx = 8 + randn() * p.cur, cy = randn() * p.cur;
      for (let t = 0; t < 22 && !out; t += dt) {
        let h = t < 2 ? 0 : Math.min(150, Math.max(0, Math.atan2(tgt[0] - ex, tgt[1] - ey) * 180 / Math.PI));
        const he = (h + bias + randn() * p.hdg * Math.sqrt(dt)) * Math.PI / 180, sp = 48 * (1 + 0.12 * randn());
        tx += (sp * Math.sin(he) + cx) * dt; ty += (sp * Math.cos(he) + cy) * dt;
        ex += 48 * Math.sin(h * Math.PI / 180) * dt; ey += 48 * Math.cos(h * Math.PI / 180) * dt;
        for (const s of sights) if (Math.abs(t - s) < dt / 2) { ex = tx + randn() * p.sig; ey = ty + randn() * p.sig; }
        tr.push([tx, ty]);
        if (inPoly(tx, ty, G.SG)) out = 1;
        else { const ay = NW[1] + (tx - NW[0]) * (SE[1] - NW[1]) / (SE[0] - NW[0]);
          if (tx > SE[0] + 10 || (tx > NW[0] + 10 && ty > ay + 25)) out = ty > ay ? 2 : 3; }
      }
      res[out]++; tracks.push([tr, out]);
    }
    draw(tracks, tgt);
    $(".cd-out").innerHTML = `<b>Landfall ${(res[1] / n * 100).toFixed(0)}%</b> · passed north ${(res[2] / n * 100).toFixed(0)}% · passed south ${(res[3] / n * 100).toFixed(0)}%`;
  }
  function draw(tracks, tgt) {
    const W = cv.clientWidth, H = cv.clientHeight, dpr = devicePixelRatio || 1; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const x0 = -40, x1 = 680, y0 = -30, y1 = 480, s = Math.min(W / (x1 - x0), H / (y1 - y0));
    const X = x => (x - x0) * s, Y = y => H - (y - y0) * s;
    ctx.fillStyle = "#f4f7fa"; ctx.fillRect(0, 0, W, H);
    const col = ["#888", "rgba(31,95,168,.35)", "rgba(217,98,43,.55)", "rgba(176,141,60,.55)"];
    for (const [tr, o] of tracks) { ctx.strokeStyle = col[o]; ctx.lineWidth = 0.8; ctx.beginPath(); tr.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.stroke(); }
    ctx.fillStyle = "#c9c2ae"; ctx.strokeStyle = "#6b6450"; ctx.beginPath(); G.SG.forEach(([x, y], i) => i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#000"; ctx.beginPath(); ctx.arc(X(G.PW[0]), Y(G.PW[1]), 4, 0, 7); ctx.fill(); ctx.font = "12px sans-serif"; ctx.fillText("Elephant Island", X(G.PW[0]) + 8, Y(G.PW[1]) - 4);
    ctx.fillText("South Georgia", X(G.SG[0][0]) - 40, Y(G.SG[0][1]) - 14);
    ctx.fillStyle = "#d9622b"; ctx.beginPath(); ctx.arc(X(tgt[0]), Y(tgt[1]), 4, 0, 7); ctx.fill();
  }
  host.querySelectorAll("[data-k]").forEach(i => { const o = i.parentElement.querySelector("output"); const f = () => o.textContent = i.value; f(); i.oninput = f; });
  $(".cd-run").onclick = run;
  new ResizeObserver(() => run()).observe(cv);
}
