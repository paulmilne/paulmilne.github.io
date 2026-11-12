// searchgame.js: Bayesian search for the wreck. Each dive searches a 3 km x 3 km block and finds
// the wreck with probability 0.9 if it is there; the map shows the posterior probability.
export function mountSearch(host) {
  host.innerHTML = `<div class="sg-grid"><canvas class="sg-map"></canvas><div class="sg-side">
    <div class="sg-row"><label><input type="radio" name="sgw" value="prior" checked> Wreck drawn from the prior</label>
    <label><input type="radio" name="sgw" value="real"> The real wreck (9.4 km south-east)</label></div>
    <label>Fix uncertainty <select class="sg-fix"><option value="5.2" selected>two fixes, σ 5.2 km</option><option value="10.1">one fix, σ 10.1 km</option></select></label>
    <div class="sg-row"><button class="sg-best">Dive on the most probable block</button><button class="sg-new">New search</button></div>
    <div class="sg-out"></div><div class="sg-note">Click a block to send the AUV there.</div></div></div>`;
  const cv = host.querySelector("canvas"), ctx = cv.getContext("2d"), $ = q => host.querySelector(q);
  const N = 39, B = 3, NB = N / B, PD = 0.9;
  let p, wreck, dives, found;
  function reset() {
    const s = +$(".sg-fix").value; p = new Float64Array(N * N); let tot = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const x = j - 19, y = 19 - i; const v = Math.exp(-(x * x + y * y) / (2 * s * s)); p[i * N + j] = v; tot += v; }
    for (let k = 0; k < p.length; k++) p[k] /= tot;
    if (host.querySelector("input[name=sgw]:checked").value === "real") wreck = [19 + 7, 19 + 7];   // row (south), col (east)
    else { let r = Math.random(), k = 0; while (k < p.length - 1 && (r -= p[k]) > 0) k++; wreck = [Math.floor(k / N), k % N]; }
    dives = 0; found = false; draw(); status();
  }
  const blockMass = (bi, bj) => { let m = 0; for (let i = bi * B; i < bi * B + B; i++) for (let j = bj * B; j < bj * B + B; j++) m += p[i * N + j]; return m; };
  function dive(bi, bj) {
    if (found) return; dives++;
    const inside = Math.floor(wreck[0] / B) === bi && Math.floor(wreck[1] / B) === bj;
    if (inside && Math.random() < PD) { found = true; draw(); status(); return; }
    const m = blockMass(bi, bj);
    for (let i = bi * B; i < bi * B + B; i++) for (let j = bj * B; j < bj * B + B; j++) p[i * N + j] *= 1 - PD;
    for (let k = 0; k < p.length; k++) p[k] /= 1 - PD * m;
    draw(); status();
  }
  function best() { let bm = -1, bb = [0, 0]; for (let bi = 0; bi < NB; bi++) for (let bj = 0; bj < NB; bj++) { const m = blockMass(bi, bj); if (m > bm) { bm = m; bb = [bi, bj]; } } dive(...bb); }
  function status() {
    $(".sg-out").innerHTML = found ? `<b>Found after ${dives} dive${dives > 1 ? "s" : ""}.</b>` : `${dives} dives so far. Most probable block holds ${(Math.max(...Array.from({ length: NB * NB }, (_, k) => blockMass(Math.floor(k / NB), k % NB))) * 100).toFixed(1)}% of the probability.`;
  }
  function draw() {
    const W = cv.clientWidth, dpr = devicePixelRatio || 1; cv.width = W * dpr; cv.height = W * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const c = W / N; let mx = 0; for (const v of p) mx = Math.max(mx, v);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const v = Math.pow(p[i * N + j] / mx, 0.6);
      ctx.fillStyle = `rgb(${Math.round(247 - 216 * v)},${Math.round(251 - 156 * v)},${Math.round(255 - 87 * v)})`; ctx.fillRect(j * c, i * c, c + 0.5, c + 0.5); }
    ctx.strokeStyle = "rgba(0,0,0,.12)"; for (let k = 0; k <= NB; k++) { ctx.beginPath(); ctx.moveTo(k * B * c, 0); ctx.lineTo(k * B * c, W); ctx.moveTo(0, k * B * c); ctx.lineTo(W, k * B * c); ctx.stroke(); }
    ctx.strokeStyle = "#000"; ctx.beginPath(); ctx.moveTo(19.5 * c - 6, 19.5 * c); ctx.lineTo(19.5 * c + 6, 19.5 * c); ctx.moveTo(19.5 * c, 19.5 * c - 6); ctx.lineTo(19.5 * c, 19.5 * c + 6); ctx.stroke();
    if (found) { ctx.strokeStyle = "#d9622b"; ctx.lineWidth = 3; const x = (wreck[1] + .5) * c, y = (wreck[0] + .5) * c; ctx.beginPath(); ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7); ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7); ctx.stroke(); ctx.lineWidth = 1; }
  }
  cv.addEventListener("click", e => { const r = cv.getBoundingClientRect(), c = r.width / N; dive(Math.min(NB - 1, Math.floor((e.clientY - r.top) / c / B)), Math.min(NB - 1, Math.floor((e.clientX - r.left) / c / B))); });
  $(".sg-best").onclick = best; $(".sg-new").onclick = reset; $(".sg-fix").onchange = reset;
  host.querySelectorAll("input[name=sgw]").forEach(r => r.onchange = reset);
  new ResizeObserver(() => draw()).observe(cv);
  reset();
}
