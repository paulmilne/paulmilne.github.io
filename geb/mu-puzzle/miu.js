// The MIU system: apply the rules by hand, and ask for a derivation of any string.
// Strings are stored without the leading M.

export const RULES = {
  1: "xI → xIU",
  2: "Mx → Mxx",
  3: "III → U",
  4: "UU → (deleted)",
};
const ROMAN = { 1: "I", 2: "II", 3: "III", 4: "IV" };
const MAXLEN = 64;

export function successors(s) {
  const out = [];
  if (s.endsWith("I")) out.push({ rule: 1, pos: s.length, len: 0, t: s + "U" });
  out.push({ rule: 2, pos: 0, len: s.length, t: s + s });
  for (let i = 0; i + 3 <= s.length; i++)
    if (s.slice(i, i + 3) === "III") out.push({ rule: 3, pos: i, len: 3, t: s.slice(0, i) + "U" + s.slice(i + 3) });
  for (let i = 0; i + 2 <= s.length; i++)
    if (s.slice(i, i + 2) === "UU") out.push({ rule: 4, pos: i, len: 2, t: s.slice(0, i) + s.slice(i + 2) });
  return out;
}

const countI = (s) => [...s].filter((c) => c === "I").length;

// A derivation of Mx when #I(x) is not a multiple of 3 (the recipe in the post).
export function construct(x) {
  const c = countI(x), u = x.length - c;
  if (c % 3 === 0) return null;
  const steps = [];
  let s = "I";
  while (!(s.length >= c + 3 * u && s.length % 3 === c % 3)) { s = s + s; steps.push({ rule: 2, t: s }); }
  let r = (s.length - c - 3 * u) / 3;
  if (r % 2) { s = s + "U"; steps.push({ rule: 1, t: s }); r++; }
  let pos = 0;
  for (const ch of x) {
    if (ch === "U") { s = s.slice(0, pos) + "U" + s.slice(pos + 3); steps.push({ rule: 3, t: s }); }
    pos++;
  }
  while (s.slice(pos, pos + 3) === "III") { s = s.slice(0, pos) + "U" + s.slice(pos + 3); steps.push({ rule: 3, t: s }); pos++; }
  while (s.endsWith("UU") && s.length > x.length) { s = s.slice(0, -2); steps.push({ rule: 4, t: s }); }
  return steps;
}

// Shortest derivation by breadth-first search, never passing through strings
// longer than cap. Returns null if not found within the state budget.
export function shortest(x, cap, budget = 400000) {
  const parent = new Map([["I", null]]);
  let frontier = ["I"];
  while (frontier.length && parent.size < budget) {
    const next = [];
    for (const s of frontier) {
      if (s === x) return unwind(parent, x);
      for (const m of successors(s)) {
        if (m.t.length > cap || parent.has(m.t)) continue;
        parent.set(m.t, { s, rule: m.rule });
        next.push(m.t);
      }
    }
    frontier = next;
  }
  return parent.has(x) ? unwind(parent, x) : null;
}
function unwind(parent, x) {
  const steps = [];
  for (let t = x; parent.get(t); t = parent.get(t).s) steps.push({ rule: parent.get(t).rule, t });
  return steps.reverse();
}

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}

// Show "M" + s, with the part a move acts on highlighted.
function marked(s, m) {
  const span = el("span", { class: "miu-str" }, "M");
  if (!m || m.rule === 2) { span.append(s); return span; }
  if (m.rule === 1) { span.append(s.slice(0, -1), el("mark", {}, "I")); return span; }
  span.append(s.slice(0, m.pos), el("mark", {}, s.slice(m.pos, m.pos + m.len)), s.slice(m.pos + m.len));
  return span;
}

export function mountMIU(host) {
  host.innerHTML = "";
  let hist = ["I"];

  // --- part 1: play ---
  const cur = el("div", { class: "miu-cur" });
  const inv = el("div", { class: "miu-inv" });
  const moves = el("div", { class: "miu-moves" });
  const trail = el("div", { class: "miu-trail" });
  const undo = el("button", { onclick: () => { if (hist.length > 1) { hist.pop(); draw(); } } }, "Undo");
  const reset = el("button", { onclick: () => { hist = ["I"]; draw(); } }, "Start again");

  function draw() {
    const s = hist[hist.length - 1];
    cur.replaceChildren(el("span", { class: "miu-str miu-big" }, "M" + s));
    const n = countI(s);
    inv.replaceChildren(
      `${s.length + 1} symbols · ${n} I${n === 1 ? "" : "'s"} · `,
      el("b", {}, `${n} mod 3 = ${n % 3}`),
      s === "U" ? el("span", { class: "miu-win" }, "  MU! (this should be impossible)") : ""
    );
    moves.replaceChildren();
    for (const m of successors(s)) {
      const big = m.t.length > MAXLEN;
      const b = el("button", { class: `miu-move r${m.rule}`, title: big ? "too long to show" : "", onclick: () => { hist.push(m.t); draw(); } },
        el("span", { class: "miu-rule" }, `${ROMAN[m.rule]}`), marked(s, m), " → ", el("span", { class: "miu-str" }, "M" + (big ? m.t.slice(0, 24) + "…" : m.t)));
      if (big) b.disabled = true;
      moves.append(b);
    }
    trail.replaceChildren(...hist.map((t, i) => el("span", {}, (i ? " → " : "") + "M" + (t.length > 20 ? t.slice(0, 18) + "…" : t))));
    undo.disabled = hist.length < 2;
  }

  // --- part 2: ask ---
  const inp = el("input", { type: "text", value: "UUUUUUI", size: 14, spellcheck: "false", class: "miu-in" });
  const out = el("div", { class: "miu-out" });
  const go = el("button", { class: "miu-go", onclick: solve }, "Derive it");
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") solve(); });

  function listSteps(steps) {
    const ol = el("ol", { class: "miu-steps" });
    ol.append(el("li", { value: 0 }, el("span", { class: "miu-str" }, "MI"), el("span", { class: "miu-why" }, " axiom")));
    steps.forEach((st, i) => ol.append(el("li", { value: i + 1 }, el("span", { class: "miu-str" }, "M" + st.t), el("span", { class: "miu-why" }, ` rule ${ROMAN[st.rule]}`))));
    return ol;
  }
  function solve() {
    let x = inp.value.toUpperCase().replace(/\s/g, "");
    if (x.startsWith("M")) x = x.slice(1);
    if (!/^[IU]+$/.test(x) || x.length > 16) {
      out.replaceChildren("Type M followed by up to 16 I's and U's.");
      return;
    }
    const n = countI(x);
    if (n % 3 === 0) {
      out.replaceChildren(el("p", {}, el("b", {}, `M${x} is not a theorem. `),
        `It has ${n} I${n === 1 ? "" : "'s"}, a multiple of 3, and every theorem's count of I's is 1 or 2 mod 3.`));
      return;
    }
    const rec = construct(x);
    out.replaceChildren(el("p", {}, "Searching…"));
    setTimeout(() => {
      const sh = x.length <= 9 ? shortest(x, Math.max(18, 2 * x.length + 4)) : null;
      const cols = el("div", { class: "miu-cols" });
      if (sh) cols.append(el("div", {}, el("div", { class: "miu-h" }, `Shortest: ${sh.length} steps`), listSteps(sh)));
      cols.append(el("div", {}, el("div", { class: "miu-h" }, `The recipe: ${rec.length} steps`), listSteps(rec)));
      out.replaceChildren(
        el("p", {}, el("b", {}, `M${x} is a theorem`), ` (${n} I${n === 1 ? "" : "'s"}, ${n % 3} mod 3).` + (sh ? "" : " Strings this long are too slow to search for the shortest route here, so only the recipe is shown.")),
        cols
      );
    }, 20);
  }

  host.append(
    el("div", { class: "miu-box" },
      el("div", { class: "miu-head" }, "Start from MI and try to reach MU."),
      cur, inv,
      el("div", { class: "miu-label" }, "Moves available (click one):"),
      moves,
      el("div", { class: "miu-row" }, undo, reset),
      el("div", { class: "miu-label" }, "Your derivation:"),
      trail
    ),
    el("div", { class: "miu-box" },
      el("div", { class: "miu-head" }, "Or type a string and ask for a derivation:"),
      el("div", { class: "miu-row" }, el("span", { class: "miu-str" }, "M"), inp, go),
      out
    )
  );
  draw();
  solve();
  return host;
}
