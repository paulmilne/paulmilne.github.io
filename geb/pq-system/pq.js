// The pq-system: choose meanings for p and q and test them against the theorems;
// type a string to see whether it is a theorem and how it is derived.

const P_OPTS = {
  "plus": { kind: "op", f: (a, b) => a + b },
  "minus": { kind: "op", f: (a, b) => a - b },
  "times": { kind: "op", f: (a, b) => a * b },
  "equals": { kind: "rel", f: (a, b) => a === b },
  "is at least": { kind: "rel", f: (a, b) => a >= b },
};
const Q_OPTS = {
  "equals": { kind: "rel", f: (a, b) => a === b },
  "is at least": { kind: "rel", f: (a, b) => a >= b },
  "is at most": { kind: "rel", f: (a, b) => a <= b },
  "plus": { kind: "op", f: (a, b) => a + b },
  "taken from": { kind: "op", f: (a, b) => b - a },   // "y taken from z" = z - y
};

// truth of "x P y Q z" under a reading; null if it is not a statement
function truth(P, Q, x, y, z) {
  const p = P_OPTS[P], q = Q_OPTS[Q];
  if (p.kind === "op" && q.kind === "rel") return q.f(p.f(x, y), z);   // (x P y) Q z
  if (p.kind === "rel" && q.kind === "op") return p.f(x, q.f(y, z));   // x P (y Q z)
  if (p.kind === "rel" && q.kind === "rel") return p.f(x, y) && q.f(y, z);
  return null;                                                          // two operations: no claim
}
const isThm = (x, y, z) => x >= 1 && y >= 1 && z === x + y;
const str = (x, y, z) => "-".repeat(x) + "p" + "-".repeat(y) + "q" + "-".repeat(z);

function parse(s) {
  const m = /^(-+)p(-+)q(-+)$/.exec(s.trim());
  return m ? [m[1].length, m[2].length, m[3].length] : null;
}
function derivation(x, y) {
  const out = [{ s: str(x, 1, x + 1), why: "axiom" }];
  for (let k = 2; k <= y; k++) out.push({ s: str(x, k, x + k), why: "rule" });
  return out;
}

export function mountPQ(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const sel = (opts, val) => { const s = el("select"); for (const k of Object.keys(opts)) { const o = el("option", null, k); o.value = k; if (k === val) o.selected = true; s.append(o); } return s; };

  // --- reading tester ---
  const box1 = el("div", "pq-box");
  const pS = sel(P_OPTS, "plus"), qS = sel(Q_OPTS, "equals");
  const head = el("div", "pq-read");
  head.append("Read ", el("code", null, "p"), " as ", pS, " and ", el("code", null, "q"), " as ", qS);
  const verdict = el("div", "pq-verdict");
  const table = el("div", "pq-table");
  box1.append(el("div", "pq-h", "Give the symbols a meaning"), head, verdict, table);

  function drawReading() {
    const P = pS.value, Q = qS.value;
    const thms = [], non = [];
    for (let x = 1; x <= 5; x++) for (let y = 1; y <= 5; y++) {
      thms.push([x, y, x + y]);
      for (const d of [-2, -1, 1, 2]) if (x + y + d >= 1) non.push([x, y, x + y + d]);
    }
    const tv = thms.map((t) => truth(P, Q, ...t)), nv = non.map((t) => truth(P, Q, ...t));
    if (tv[0] === null) {
      verdict.innerHTML = `<span class="pq-no">With both symbols read as operations, a string makes no claim at all, so it cannot be true or false.</span>`;
    } else {
      const tTrue = tv.filter(Boolean).length, nTrue = nv.filter(Boolean).length;
      let msg;
      if (tTrue === thms.length && nTrue === 0) msg = `<span class="pq-ok">An isomorphism.</span> All ${thms.length} theorems shown are true, and all ${non.length} near-miss non-theorems are false.`;
      else if (tTrue === thms.length) msg = `<span class="pq-half">True of every theorem, but not only of theorems:</span> ${nTrue} of ${non.length} non-theorems are true as well. The reading is consistent with the system but does not capture it.`;
      else msg = `<span class="pq-no">Not a faithful reading:</span> ${tTrue === 0 ? "none" : "only " + tTrue} of the ${thms.length} theorems ${tTrue === 1 ? "comes" : "come"} out true.`;
      verdict.innerHTML = msg;
    }
    table.replaceChildren();
    const show = thms.slice(0, 12).map((t) => [t, true]).concat(non.filter((_, i) => i % 7 === 0).slice(0, 6).map((t) => [t, false]));
    for (const [[x, y, z], th] of show) {
      const v = truth(P, Q, x, y, z);
      const row = el("div", "pq-row" + (th ? "" : " pq-non"));
      row.append(el("code", null, str(x, y, z)),
        el("span", "pq-tag", th ? "theorem" : "not a theorem"),
        el("span", "pq-say", `${x} ${P} ${y} ${Q} ${z}`),
        el("span", v === null ? "pq-dim" : v ? "pq-t" : "pq-f", v === null ? "—" : v ? "true" : "false"));
      table.append(row);
    }
  }
  pS.onchange = qS.onchange = drawReading;

  // --- string checker ---
  const box2 = el("div", "pq-box");
  const inp = el("input", "pq-in"); inp.value = "--p---q-----"; inp.spellcheck = false;
  const btn = el("button", "pq-go", "Check");
  const res = el("div", "pq-res");
  const row = el("div", "pq-row2"); row.append(inp, btn);
  box2.append(el("div", "pq-h", "Check a string"), row, res);
  function check() {
    const s = inp.value.trim(), t = parse(s);
    if (!t) { res.innerHTML = `<p><b>Not well formed.</b> A pq-string is some hyphens, p, some hyphens, q, some hyphens.</p>`; return; }
    const [x, y, z] = t;
    if (!isThm(x, y, z)) {
      res.innerHTML = `<p><b>Not a theorem.</b> ${x} + ${y} = ${x + y}, not ${z}. Every rule keeps the last group exactly as long as the first two together.</p>`;
      return;
    }
    const d = derivation(x, y);
    res.innerHTML = `<p><b>A theorem</b>, derived in ${d.length} step${d.length > 1 ? "s" : ""}. Read as addition: ${x} + ${y} = ${z}. Read as subtraction: ${x} = ${z} − ${y}.</p>`;
    const ol = el("ol", "pq-steps");
    d.forEach((st) => { const li = el("li"); li.append(el("code", null, st.s), el("span", "pq-dim", "  " + st.why)); ol.append(li); });
    res.append(ol);
  }
  btn.onclick = check; inp.addEventListener("keydown", (e) => { if (e.key === "Enter") check(); });

  host.append(box1, box2);
  drawReading(); check();
  return host;
}
