// watch.js: a second hand that twitches in Morse, read back by a decoder with timing noise.
const M = {A:".-",B:"-...",C:"-.-.",D:"-..",E:".",F:"..-.",G:"--.",H:"....",I:"..",J:".---",K:"-.-",L:".-..",M:"--",N:"-.",O:"---",P:".--.",Q:"--.-",R:".-.",S:"...",T:"-",U:"..-",V:"...-",W:".--",X:"-..-",Y:"-.--",Z:"--..",
  "0":"-----","1":".----","2":"..---","3":"...--","4":"....-","5":".....","6":"-....","7":"--...","8":"---..","9":"----."};
const R = Object.fromEntries(Object.entries(M).map(([k, v]) => [v, k]));
export function encode(text) {
  const seg = [], words = text.toUpperCase().replace(/[^A-Z0-9 ]/g, "").split(/\s+/).filter(Boolean);
  words.forEach((w, wi) => { [...w].forEach((ch, li) => { const c = M[ch];
      [...c].forEach((e, ei) => { seg.push([1, e === "." ? 1 : 3, ch]); if (ei < c.length - 1) seg.push([0, 1]); });
      if (li < w.length - 1) seg.push([0, 3]); });
    if (wi < words.length - 1) seg.push([0, 7]); });
  return seg;
}
export function mountWatch(host) {
  host.innerHTML = `<div class="wt-grid"><svg class="wt-face" viewBox="-110 -110 220 220"></svg>
   <div class="wt-side"><label>Message <input class="wt-msg" value="STAY"></label>
   <label>Unit <input type="range" class="wt-unit" min="0.08" max="1" step="0.01" value="0.25"><output></output></label>
   <label>Timing jitter σ <input type="range" class="wt-jit" min="0" max="0.5" step="0.01" value="0.05"><output></output></label>
   <div class="wt-btns"><button class="wt-go">Send</button></div>
   <div class="wt-out"><div><b>Sent</b> <span class="wt-sent"></span></div><div><b>Morse</b> <span class="wt-code"></span></div><div><b>Read</b> <span class="wt-read"></span></div><div class="wt-rate"></div></div></div></div>`;
  const svg = host.querySelector("svg"), NS = "http://www.w3.org/2000/svg";
  const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
  el("circle", { r: 100, fill: "#f7f4ec", stroke: "#3b3b3b", "stroke-width": 5 });
  for (let i = 0; i < 60; i++) { const a = i * Math.PI / 30, r1 = i % 5 ? 90 : 82;
    el("line", { x1: r1 * Math.sin(a), y1: -r1 * Math.cos(a), x2: 95 * Math.sin(a), y2: -95 * Math.cos(a), stroke: "#333", "stroke-width": i % 5 ? 1 : 2.5 }); }
  const hr = el("line", { x1: 0, y1: 0, x2: 0, y2: -48, stroke: "#222", "stroke-width": 6, "stroke-linecap": "round", transform: "rotate(300)" });
  const mn = el("line", { x1: 0, y1: 0, x2: 0, y2: -72, stroke: "#222", "stroke-width": 4, "stroke-linecap": "round", transform: "rotate(48)" });
  const sec = el("line", { x1: 0, y1: 16, x2: 0, y2: -88, stroke: "#c0392b", "stroke-width": 2, "stroke-linecap": "round" });
  el("circle", { r: 4, fill: "#c0392b" });
  const $ = q => host.querySelector(q);
  const outs = () => { $(".wt-unit + output").textContent = (+$(".wt-unit").value).toFixed(2) + " s"; $(".wt-jit + output").textContent = (+$(".wt-jit").value).toFixed(2); };
  $(".wt-unit").oninput = outs; $(".wt-jit").oninput = outs; outs();
  let base = 0, defl = 0, run = null, last = performance.now(), tickT = 0;
  function start() {
    const text = $(".wt-msg").value, seg = encode(text), unit = +$(".wt-unit").value, jit = +$(".wt-jit").value;
    // the reader measures each duration with Gaussian error sigma*sqrt(d) (in units)
    const g = () => { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const meas = seg.map(([on, d]) => Math.max(0.05, d + jit * Math.sqrt(d) * g()));
    run = { seg, meas, unit, k: 0, t: 0, cur: "", read: "", code: "", sent: text.toUpperCase().replace(/[^A-Z0-9 ]/g, "").trim() };
    $(".wt-sent").textContent = run.sent; $(".wt-read").textContent = ""; $(".wt-code").textContent = "";
    const units = seg.reduce((s, x) => s + x[1], 0);
    $(".wt-rate").textContent = `${units} units, ${(units * unit).toFixed(1)} s for ${run.sent.replace(/ /g, "").length} characters`;
  }
  $(".wt-go").onclick = start;
  function step(now) {
    if (!host.isConnected) return;
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (run && run.k < run.seg.length) {
      run.t += dt / run.unit;
      const [on, d] = run.seg[run.k];
      defl = on ? 12 : 0;
      if (run.t >= d) {
        const m = run.meas[run.k];
        if (on) { run.cur += m < 2 ? "." : "-"; run.code += d === 1 ? "." : "-"; }
        else if (m >= 2) { run.read += R[run.cur] || "?"; run.cur = ""; run.code += m >= 5 ? " / " : " "; if (m >= 5) run.read += " "; }
        run.t -= d; run.k++;
        if (run.k === run.seg.length) { run.read += R[run.cur] || "?"; defl = 0; }
        $(".wt-read").textContent = run.read; $(".wt-code").textContent = run.code;
        const ok = run.read.replace(/ /g, "") === run.sent.replace(/ /g, "").slice(0, run.read.replace(/ /g, "").length);
        $(".wt-read").className = "wt-read " + (ok ? "" : "bad");
      }
    } else { tickT += dt; if (tickT >= 1) { tickT -= 1; base = (base + 6) % 360; } defl = 0; }
    sec.setAttribute("transform", `rotate(${base + defl})`);
    requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}
