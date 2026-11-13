// Is this a topology? Choose which subsets of X = {a, b, c} are open, and the page checks the
// axioms: the empty set and X are open, unions of open sets are open, intersections of open
// sets are open (on a finite set, finite intersections are all there are). It also tracks how
// many of the 29 topologies on three points you have found.
export function mountTopo(host) {
  host.innerHTML = "";
  const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
  const names = ["a", "b", "c"], X = 7;
  const show = (m) => m === 0 ? "∅" : "{" + names.filter((_, i) => m >> i & 1).join(", ") + "}";
  const subsets = [1, 2, 4, 3, 5, 6];                       // proper, non-empty
  const chosen = new Set([0, X]); const found = new Set();
  const wrap = el("div", "tp-wrap"), left = el("div", "tp-left"), right = el("div", "tp-right");
  left.append(el("div", "tp-head", "Open sets (tick to include):"));
  const fixed = el("div", "tp-fixed", "∅ and {a, b, c} are always open"); left.append(fixed);
  const boxes = {};
  for (const m of subsets) { const lab = el("label", "tp-box"), cb = el("input"); cb.type = "checkbox"; cb.onchange = () => { cb.checked ? chosen.add(m) : chosen.delete(m); update(); };
    boxes[m] = cb; lab.append(cb, document.createTextNode(" " + show(m))); left.append(lab); }
  const presets = el("div", "tp-presets");
  const preset = (label, ms) => { const b = el("button", "tp-btn", label); b.onclick = () => { for (const m of subsets) { boxes[m].checked = ms.includes(m); ms.includes(m) ? chosen.add(m) : chosen.delete(m); } update(); }; presets.append(b); };
  preset("indiscrete", []); preset("discrete", subsets); preset("nested", [1, 3]); preset("not a topology", [1, 2]);
  left.append(presets);
  const out = el("div", "tp-out"); right.append(out); wrap.append(left, right); host.append(wrap);
  function update() {
    const T = [...chosen]; const missing = new Map();
    for (const A of T) for (const B of T) {
      if (!chosen.has(A | B)) missing.set(A | B, `${show(A)} ∪ ${show(B)} = ${show(A | B)}`);
      if (!chosen.has(A & B)) missing.set(A & B, `${show(A)} ∩ ${show(B)} = ${show(A & B)}`);
    }
    out.innerHTML = "";
    const list = T.sort((p, q) => (p.toString(2).split("1").length - q.toString(2).split("1").length) || p - q).map(show).join(",  ");
    out.append(el("div", "tp-list", "Open sets: " + list));
    if (missing.size === 0) {
      const key = [...chosen].sort((p, q) => p - q).join(","); found.add(key);
      const ok = el("div", "tp-ok", "✓ This is a topology on {a, b, c}."); out.append(ok);
      const separated = (x, y) => T.some((U) => T.some((V) => (U >> x & 1) && (V >> y & 1) && (U & V) === 0));
      const pairs = [[0, 1], [0, 2], [1, 2]], haus = pairs.every(([x, y]) => separated(x, y));
      out.append(el("div", null, haus ? "It is Hausdorff: any two points sit in disjoint open sets. On a finite set, only the discrete topology manages this."
        : "It is not Hausdorff: some pair of points cannot be put in disjoint open sets."));
    } else {
      out.append(el("div", "tp-no", "✗ Not a topology. These sets would also have to be open:"));
      const ul = el("ul"); for (const s of missing.values()) ul.append(el("li", null, s)); out.append(ul);
    }
    out.append(el("div", "tp-count", `Different topologies you have found: ${found.size} of 29.`));
  }
  update(); return host;
}
