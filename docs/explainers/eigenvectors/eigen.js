// Interactive 2x2 eigenvector explorer.
// Edit the matrix, drag the blue vector x, and watch Ax. Eigenvectors are the
// directions where x and Ax line up.

const PRESETS = {
  "Symmetric": [2, 1, 1, 2],
  "Not symmetric": [1.5, 1, 0, 0.5],
  "Shear": [1, 1, 0, 1],
  "Rotation": [0.8, -1.2, 1.2, 0.8],
  "Reflection": [1, 0, 0, -1],
  "Projection": [0.5, 0.5, 0.5, 0.5],
  "Fibonacci": [1, 1, 1, 0],
  "Saddle": [1, 0.5, 0.5, -1],
};

function eig2([a, b, c, d]) {
  const T = a + d, D = a * d - b * c, disc = T * T / 4 - D;
  if (disc < -1e-12) {
    const re = T / 2, im = Math.sqrt(-disc);
    return { T, D, disc, complex: true, re, im };
  }
  const s = Math.sqrt(Math.max(disc, 0));
  const lams = [T / 2 + s, T / 2 - s];
  const vecs = lams.map((l) => {
    // (A - lI) v = 0: use whichever row is larger
    let v;
    if (Math.abs(b) + Math.abs(a - l) > Math.abs(c) + Math.abs(d - l)) v = [-b, a - l];
    else v = [d - l, -c];
    const n = Math.hypot(v[0], v[1]);
    if (n < 1e-12) return null; // A = lI: every vector is an eigenvector
    const sg = v[0] < -1e-12 || (Math.abs(v[0]) <= 1e-12 && v[1] < 0) ? -1 : 1; // point the positive way
    return [(sg * v[0]) / n, (sg * v[1]) / n];
  });
  let defective = false;
  if (disc <= 1e-12) {
    const isScalar = Math.abs(b) < 1e-12 && Math.abs(c) < 1e-12 && Math.abs(a - d) < 1e-12;
    if (isScalar) return { T, D, disc, complex: false, lams, vecs: [[1, 0], [0, 1]], everyVector: true };
    defective = true;
  }
  return { T, D, disc, complex: false, lams, vecs, defective };
}

const fmt = (x, n = 3) => {
  if (Math.abs(x) < 5e-10) x = 0;
  return (+x.toFixed(n)).toString().replace("-", "−");
};

function classify(e) {
  // continuous time x' = Ax
  let ct;
  if (e.D < -1e-9) ct = "saddle: unstable, one direction in and one out";
  else if (Math.abs(e.D) <= 1e-9) ct = "a zero eigenvalue: a whole line of equilibria";
  else if (e.complex) ct = Math.abs(e.T) < 1e-9 ? "centre: closed orbits" : e.T < 0 ? "stable spiral" : "unstable spiral";
  else ct = e.T < 0 ? "stable node" : "unstable node";
  // discrete time x_{k+1} = A x_k
  const rho = e.complex ? Math.hypot(e.re, e.im) : Math.max(...e.lams.map(Math.abs));
  const dt = rho < 1 - 1e-9 ? "shrinks to zero" : rho > 1 + 1e-9 ? "grows without limit"
    : e.defective ? "still grows, but only in proportion to the number of steps, because A is defective"
    : "neither grows nor shrinks overall";
  return { ct, dt, rho };
}

export function mountEigen(host) {
  host.innerHTML = "";
  const DPR = window.devicePixelRatio || 1;
  let M = [...PRESETS["Symmetric"]];
  let x = [Math.cos(0.5), Math.sin(0.5)];
  let trail = [];

  const wrap = document.createElement("div"); wrap.className = "ev-grid";
  const cv = document.createElement("canvas"); cv.className = "ev-canvas";
  const side = document.createElement("div"); side.className = "ev-side";
  wrap.append(cv, side); host.append(wrap);

  // matrix inputs
  const mat = document.createElement("div"); mat.className = "ev-mat";
  const inputs = M.map((v, i) => {
    const inp = document.createElement("input");
    inp.type = "number"; inp.step = "0.1"; inp.min = "-5"; inp.max = "5"; inp.value = v;
    inp.addEventListener("input", () => { const n = parseFloat(inp.value); if (!isNaN(n)) { M[i] = n; trail = []; draw(); } });
    return inp;
  });
  mat.append(Object.assign(document.createElement("span"), { className: "ev-A", textContent: "A =" }));
  const box = document.createElement("div"); box.className = "ev-box"; box.append(...inputs);
  mat.append(box);
  const presets = document.createElement("div"); presets.className = "ev-presets";
  for (const [name, v] of Object.entries(PRESETS)) {
    const b = document.createElement("button"); b.textContent = name;
    b.onclick = () => { M = [...v]; inputs.forEach((inp, i) => (inp.value = v[i])); trail = []; draw(); };
    presets.append(b);
  }
  const iter = document.createElement("div"); iter.className = "ev-presets";
  const bApply = document.createElement("button"); bApply.textContent = "Apply A to x";
  bApply.onclick = () => {
    const y = [M[0] * x[0] + M[1] * x[1], M[2] * x[0] + M[3] * x[1]];
    const n = Math.hypot(...y); if (n < 1e-9) return;
    trail.push(x); x = [y[0] / n, y[1] / n]; draw();
  };
  const bClear = document.createElement("button"); bClear.textContent = "Clear trail";
  bClear.onclick = () => { trail = []; draw(); };
  iter.append(bApply, bClear);
  const info = document.createElement("div"); info.className = "ev-info";
  side.append(mat, presets,
    Object.assign(document.createElement("div"), { className: "ev-note", textContent: "Drag the blue vector x. Press “Apply A to x” repeatedly: x is replaced by Ax, rescaled to length 1." }),
    iter, info);

  let W = 0, S = 0;
  const R = 3.2; // half-width in model units
  const toPx = ([u, v]) => [W / 2 + (u / R) * (W / 2), W / 2 - (v / R) * (W / 2)];
  const fromPx = (px, py) => [((px - W / 2) / (W / 2)) * R, (-(py - W / 2) / (W / 2)) * R];

  function resize() {
    W = Math.min(cv.parentElement.clientWidth, 460);
    if (wrap.clientWidth > 640) W = Math.min(460, wrap.clientWidth * 0.55);
    cv.style.width = W + "px"; cv.style.height = W + "px";
    cv.width = W * DPR; cv.height = W * DPR;
    draw();
  }

  function line(ctx, p, q, color, w = 1, dash = []) {
    ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.moveTo(...toPx(p)); ctx.lineTo(...toPx(q)); ctx.stroke(); ctx.setLineDash([]);
  }
  function arrow(ctx, p, q, color, w = 2, dash = []) {
    line(ctx, p, q, color, w, dash);
    const [x1, y1] = toPx(p), [x2, y2] = toPx(q);
    const ang = Math.atan2(y2 - y1, x2 - x1), h = 9;
    if (Math.hypot(x2 - x1, y2 - y1) < 3) return;
    ctx.beginPath(); ctx.fillStyle = color;
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - h * Math.cos(ang - 0.4), y2 - h * Math.sin(ang - 0.4));
    ctx.lineTo(x2 - h * Math.cos(ang + 0.4), y2 - h * Math.sin(ang + 0.4));
    ctx.fill();
  }
  const Ax = (v) => [M[0] * v[0] + M[1] * v[1], M[2] * v[0] + M[3] * v[1]];

  function draw() {
    if (!W) return;
    const ctx = cv.getContext("2d");
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    const dark = matchMedia("(prefers-color-scheme: dark)").matches && document.body.classList.contains("quarto-dark");
    ctx.fillStyle = dark ? "#1e1e1e" : "#ffffff"; ctx.fillRect(0, 0, W, W);
    const ink = dark ? "#ddd" : "#111";
    // transformed grid
    for (let k = -4; k <= 4; k += 0.5) {
      line(ctx, Ax([k, -4]), Ax([k, 4]), "#d6e0ee", 0.7);
      line(ctx, Ax([-4, k]), Ax([4, k]), "#d6e0ee", 0.7);
    }
    line(ctx, [-R, 0], [R, 0], "#aaa", 0.8); line(ctx, [0, -R], [0, R], "#aaa", 0.8);
    // unit circle and its image
    ctx.beginPath(); ctx.setLineDash([4, 4]); ctx.strokeStyle = "#888"; ctx.lineWidth = 1;
    for (let i = 0; i <= 200; i++) { const t = (i / 200) * 2 * Math.PI; const p = toPx([Math.cos(t), Math.sin(t)]); i ? ctx.lineTo(...p) : ctx.moveTo(...p); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.strokeStyle = ink; ctx.lineWidth = 1.8;
    for (let i = 0; i <= 200; i++) { const t = (i / 200) * 2 * Math.PI; const p = toPx(Ax([Math.cos(t), Math.sin(t)])); i ? ctx.lineTo(...p) : ctx.moveTo(...p); }
    ctx.stroke();
    // eigenvectors
    const e = eig2(M);
    const cols = ["#d9622b", "#2a9d6f"];
    if (!e.complex && !e.everyVector) {
      const n = e.defective ? 1 : 2;
      for (let i = 0; i < n; i++) {
        const v = e.vecs[i]; if (!v) continue;
        line(ctx, [-5 * v[0], -5 * v[1]], [5 * v[0], 5 * v[1]], cols[i], 1.2, [2, 3]);
        arrow(ctx, [0, 0], [e.lams[i] * v[0], e.lams[i] * v[1]], cols[i], 3);
      }
    }
    // trail
    trail.forEach((t, i) => arrow(ctx, [0, 0], t, `rgba(31,95,168,${0.12 + 0.5 * (i + 1) / (trail.length + 1)})`, 1.2));
    // x and Ax
    const y = Ax(x);
    const cross = x[0] * y[1] - x[1] * y[0], ny = Math.hypot(...y);
    const aligned = ny > 1e-9 && Math.abs(cross) / ny < 0.035;
    arrow(ctx, [0, 0], y, aligned ? "#c0392b" : "#7b4fa3", 2.5);
    arrow(ctx, [0, 0], x, "#1f5fa8", 2.5);
    const px = toPx(x); ctx.beginPath(); ctx.fillStyle = "#1f5fa8"; ctx.arc(...px, 6, 0, 2 * Math.PI); ctx.fill();
    ctx.font = "13px system-ui, sans-serif"; ctx.fillStyle = "#1f5fa8"; ctx.fillText("x", px[0] + 8, px[1] - 6);
    const py = toPx(y); ctx.fillStyle = aligned ? "#c0392b" : "#7b4fa3"; ctx.fillText("Ax", py[0] + 8, py[1] - 6);

    // info
    const c = classify(e);
    const rows = [];
    rows.push(`<div>trace ${fmt(e.T)} · determinant ${fmt(e.D)}</div>`);
    if (e.complex) {
      const r = Math.hypot(e.re, e.im), th = (Math.atan2(e.im, e.re) * 180) / Math.PI;
      rows.push(`<div><b>λ = ${fmt(e.re)} ± ${fmt(e.im)}i</b>: no real eigenvectors. Every direction is turned by ${fmt(th, 1)}° and scaled by |λ| = ${fmt(r)}.</div>`);
    } else if (e.everyVector) {
      rows.push(`<div><b>λ = ${fmt(e.lams[0])}</b> twice: A is a multiple of the identity, so every vector is an eigenvector.</div>`);
    } else if (e.defective) {
      rows.push(`<div><b>λ = ${fmt(e.lams[0])}</b> twice, but only <span style="color:${cols[0]}">one eigenvector</span> (${fmt(e.vecs[0][0])}, ${fmt(e.vecs[0][1])}). A cannot be diagonalised.</div>`);
    } else {
      e.lams.forEach((l, i) => rows.push(`<div style="color:${cols[i]}"><b>λ${i ? "₂" : "₁"} = ${fmt(l)}</b>, eigenvector (${fmt(e.vecs[i][0])}, ${fmt(e.vecs[i][1])})</div>`));
    }
    rows.push(`<div class="ev-dim">As ẋ = Ax: ${c.ct}.</div>`);
    rows.push(`<div class="ev-dim">Applied repeatedly: largest |λ| = ${fmt(c.rho)}, so a typical vector ${c.dt}.</div>`);
    rows.push(aligned ? `<div style="color:#c0392b"><b>x is (nearly) an eigenvector:</b> Ax = ${fmt(ny * Math.sign(x[0] * y[0] + x[1] * y[1]))} x</div>`
                      : `<div class="ev-dim">x is not an eigenvector: Ax points a different way.</div>`);
    info.innerHTML = rows.join("");
  }

  // dragging
  let drag = false;
  const pos = (ev) => { const r = cv.getBoundingClientRect(); const t = ev.touches ? ev.touches[0] : ev; return fromPx(t.clientX - r.left, t.clientY - r.top); };
  const down = (ev) => { drag = true; move(ev); };
  const move = (ev) => {
    if (!drag) return; ev.preventDefault();
    const p = pos(ev); const n = Math.hypot(...p); if (n < 0.05) return;
    x = [p[0] / n, p[1] / n]; trail = []; draw();
  };
  const up = () => (drag = false);
  cv.addEventListener("mousedown", down); window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
  cv.addEventListener("touchstart", down, { passive: false }); cv.addEventListener("touchmove", move, { passive: false }); cv.addEventListener("touchend", up);

  new ResizeObserver(resize).observe(wrap);
  resize();
  return host;
}
