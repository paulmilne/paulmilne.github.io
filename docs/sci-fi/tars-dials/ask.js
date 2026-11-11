// ask.js: interrogate a robot whose honesty is set by a dial, lying at random or by policy.
export function mountAsk(host) {
  host.innerHTML = `<div class="ak">
   <div class="ak-row"><label>Honesty <input type="range" class="ak-h" min="50" max="100" step="1" value="90"><output></output></label>
   <label class="ak-mode"><input type="radio" name="akm" value="random" checked> Lies at random</label>
   <label class="ak-mode"><input type="radio" name="akm" value="policy"> Lies by policy (decides once per topic)</label></div>
   <div class="ak-q">Hidden fact #<span class="ak-n">1</span>: "Is the door code even?"</div>
   <div class="ak-row"><button class="ak-ask">Ask</button><button class="ak-ask5">Ask 5 times</button><button class="ak-reveal">Reveal and next</button></div>
   <div class="ak-ans"></div><div class="ak-belief"></div><div class="ak-score"></div></div>`;
  const $ = q => host.querySelector(q);
  let truth, liesHere, answers, n = 1, right = 0, done = 0;
  const h = () => +$(".ak-h").value / 100, mode = () => host.querySelector("input[name=akm]:checked").value;
  function fresh() { truth = Math.random() < 0.5; liesHere = Math.random() > h(); answers = []; show(); }
  function ask() {
    const lie = mode() === "policy" ? liesHere : Math.random() > h();
    answers.push(lie ? !truth : truth); show();
  }
  function show() {
    $(".ak-h + output").textContent = Math.round(h() * 100) + "%"; $(".ak-n").textContent = n;
    $(".ak-ans").innerHTML = answers.length ? answers.map(a => `<span class="${a ? "y" : "n"}">${a ? "yes" : "no"}</span>`).join(" ") : "<i>no answers yet</i>";
    // belief if you assume each answer is independently honest with probability h
    const y = answers.filter(a => a).length, k = answers.length - y, p = h();
    if (p >= 0.999 && y && k) { $(".ak-belief").textContent = "Contradictory answers from a robot you trusted completely."; return; }
    const lr = Math.pow(p / (1 - p + 1e-12), y - k), py = lr / (1 + lr);
    $(".ak-belief").textContent = answers.length ? `If the lies are independent: P(yes) = ${(py * 100).toFixed(py > 0.999 || py < 0.001 ? 2 : 1)}%` : "";
  }
  $(".ak-ask").onclick = ask; $(".ak-ask5").onclick = () => { for (let i = 0; i < 5; i++) ask(); };
  $(".ak-reveal").onclick = () => {
    if (answers.length) { const y = answers.filter(a => a).length, guess = y * 2 === answers.length ? Math.random() < 0.5 : y * 2 > answers.length; done++; if (guess === truth) right++; }
    $(".ak-score").textContent = `The answer was ${truth ? "yes" : "no"}${mode() === "policy" ? (liesHere ? ": this was a topic it lies about." : ": a topic it is honest about.") : "."} Majority vote right ${right} of ${done}.`;
    n++; fresh();
  };
  $(".ak-h").oninput = () => { liesHere = Math.random() > h(); show(); };
  host.querySelectorAll("input[name=akm]").forEach(r => r.onchange = () => { right = 0; done = 0; $(".ak-score").textContent = ""; fresh(); });
  fresh();
}
