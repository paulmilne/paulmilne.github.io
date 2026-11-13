// Explorer for the Federal Reserve's Distributional Financial Accounts.
// Data: dfa.json, written by src/economics/part1.py.

const COLS = { "Top 0.1%": "#d9622b", "Next 0.9%": "#b08d3c", "Next 9%": "#1f5fa8", "Next 40%": "#2a9d6f", "Bottom 50%": "#7b4fa3" };

export async function mountDFA(host) {
  host.innerHTML = "";
  const D = await (await fetch(new URL("dfa.json", document.baseURI))).json();
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const years = D.quarters.map((q) => +q.slice(0, 4) + (+q.slice(5) - 1) / 4);

  const views = {
    "Share of all household net worth": { get: (g) => D.share[g], unit: "%", note: "Each group's share of total US household net worth." },
    "Real net worth, 1989 = 100": { get: (g) => D.real_index[g], unit: "", log: true, note: "Each group's total net worth after inflation, as an index. The scale is logarithmic, so equal slopes mean equal growth rates. Groups are fixed shares of households, not fixed people." },
  };
  for (const a of D.assets) views["Share of: " + a.toLowerCase()] = { get: (g) => D.asset_share[g][a], unit: "%", note: `Each group's share of all household holdings of ${a.toLowerCase()}.` };

  const sel = el("select"); for (const k of Object.keys(views)) { const o = el("option", null, k); o.value = k; sel.append(o); }
  const checks = el("div", "dfa-checks"); const on = Object.fromEntries(D.groups.map((g) => [g, true]));
  for (const g of D.groups) {
    const l = el("label"); const c = document.createElement("input"); c.type = "checkbox"; c.checked = true;
    c.onchange = () => { on[g] = c.checked; draw(); };
    const sw = el("span", "dfa-sw"); sw.style.background = COLS[g]; l.append(c, sw, g); checks.append(l);
  }
  const cv = el("canvas", "dfa-canvas"), note = el("div", "dfa-note"), read = el("div", "dfa-read");
  const top = el("div", "dfa-top"); top.append("Show ", sel);
  host.append(top, checks, cv, read, note);
  const DPR = window.devicePixelRatio || 1; let W = 0, H = 0, hover = null;
  const M = { l: 46, r: 14, t: 10, b: 26 };

  function scales() {
    const v = views[sel.value]; const series = D.groups.filter((g) => on[g]).map((g) => v.get(g));
    let lo = Math.min(...series.flat()), hi = Math.max(...series.flat());
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (v.log) { lo = Math.max(lo, 1); const a = Math.log(lo * 0.9), b = Math.log(hi * 1.1); return { v, y: (x) => M.t + (1 - (Math.log(Math.max(x, 1)) - a) / (b - a)) * (H - M.t - M.b), lo, hi }; }
    lo = Math.min(0, lo); hi = hi * 1.08;
    return { v, y: (x) => M.t + (1 - (x - lo) / (hi - lo)) * (H - M.t - M.b), lo, hi };
  }
  const X = (t) => M.l + ((t - years[0]) / (years[years.length - 1] - years[0])) * (W - M.l - M.r);

  function draw() {
    if (!W) return;
    const ctx = cv.getContext("2d"); ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
    const S = scales(); ctx.font = "11px system-ui, sans-serif";
    // grid
    const ticks = S.v.log ? [25, 50, 100, 200, 400, 800, 1600].filter((t) => t >= S.lo * 0.9 && t <= S.hi * 1.1)
      : Array.from({ length: 6 }, (_, i) => S.lo + ((S.hi - S.lo) * i) / 5);
    for (const t of ticks) { const y = S.y(t); ctx.strokeStyle = "#e6e6e6"; ctx.beginPath(); ctx.moveTo(M.l, y); ctx.lineTo(W - M.r, y); ctx.stroke(); ctx.fillStyle = "#777"; ctx.textAlign = "right"; ctx.fillText((S.v.log ? t : t.toFixed(t < 10 && S.hi < 12 ? 1 : 0)) + S.v.unit, M.l - 5, y + 4); }
    ctx.textAlign = "center";
    for (let yr = 1990; yr <= 2026; yr += 5) { ctx.fillStyle = "#777"; ctx.fillText(yr, X(yr), H - 8); }
    for (const g of D.groups) {
      if (!on[g]) continue; const s = S.v.get(g);
      ctx.strokeStyle = COLS[g]; ctx.lineWidth = g === "Top 0.1%" ? 2.4 : 1.6; ctx.beginPath();
      s.forEach((val, i) => (i ? ctx.lineTo(X(years[i]), S.y(val)) : ctx.moveTo(X(years[i]), S.y(val)))); ctx.stroke();
    }
    const i = hover == null ? years.length - 1 : hover;
    ctx.strokeStyle = "#999"; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(years[i]), M.t); ctx.lineTo(X(years[i]), H - M.b); ctx.stroke(); ctx.setLineDash([]);
    read.replaceChildren(el("b", null, D.quarters[i].replace("Q", " Q") + ":  "),
      ...D.groups.filter((g) => on[g]).map((g) => { const s = el("span", "dfa-val", `${g} ${S.v.get(g)[i].toFixed(1)}${S.v.unit}`); s.style.color = COLS[g]; return s; }));
    note.textContent = S.v.note + " Source: Federal Reserve, Distributional Financial Accounts.";
  }
  function resize() { W = host.clientWidth || 600; H = Math.min(380, Math.max(260, W * 0.5)); cv.style.width = W + "px"; cv.style.height = H + "px"; cv.width = W * DPR; cv.height = H * DPR; draw(); }
  const pick = (ev) => { const r = cv.getBoundingClientRect(); const cx = (ev.touches ? ev.touches[0] : ev).clientX - r.left;
    const t = years[0] + ((cx - M.l) / (W - M.l - M.r)) * (years[years.length - 1] - years[0]);
    let best = 0; years.forEach((y, k) => { if (Math.abs(y - t) < Math.abs(years[best] - t)) best = k; }); hover = best; draw(); };
  cv.addEventListener("mousemove", pick); cv.addEventListener("touchmove", pick, { passive: true }); cv.addEventListener("mouseleave", () => { hover = null; draw(); });
  sel.onchange = draw;
  new ResizeObserver(resize).observe(host); resize();
  return host;
}
