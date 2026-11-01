"""Word-level vocabulary as a variety ledger: Heaps discovery curve, top-K residual on held-out text, Good-Turing missing mass."""
import json, collections, numpy as np
text = open("/home/claude/emb/dl/text8").read()
tr = text[:95_000_000].split()[:-1]; ho = text[95_000_000:].split()
# Heaps
ns = np.unique(np.logspace(2, np.log10(len(tr)), 60).astype(int)); seen = set(); D = []; j = 0
for i, x in enumerate(tr, 1):
    seen.add(x)
    while j < len(ns) and i == ns[j]: D.append(len(seen)); j += 1
ns = ns[:len(D)]
fit = lambda m: float(np.polyfit(np.log(ns[m]), np.log(np.array(D)[m]), 1)[0])
beta_all = fit(ns >= 1e3); beta_hi = fit(ns >= 1e6)
# top-K residual on held-out
c = collections.Counter(tr); hc = collections.Counter(ho); Hn = sum(hc.values())
ranked = [w for w, _ in c.most_common()]
rank = {w: i + 1 for i, w in enumerate(ranked)}
hr = sorted((rank.get(w, 10**9), v) for w, v in hc.items())
Ks = np.unique(np.logspace(1, np.log10(len(ranked)), 50).astype(int)); res = []; cum = 0; i = 0
for K in Ks:
    while i < len(hr) and hr[i][0] <= K: cum += hr[i][1]; i += 1
    res.append([int(K), 1 - cum / Hn])
f = np.array(sorted(c.values(), reverse=True), float); r = np.arange(1, len(f) + 1); m = (r >= 100) & (r <= 100000)
alpha = float(-np.polyfit(np.log(r[m]), np.log(f[m]), 1)[0])
loc = lambda a, b: float(np.log(dict(res)[a] / dict(res)[b]) / np.log(b / a)) if a in dict(res) and b in dict(res) else None
Rk = dict(res)
def near(K): return min(Rk, key=lambda k: abs(np.log(k / K)))
slopes = [[near(a), near(b), float(np.log(Rk[near(a)] / Rk[near(b)]) / np.log(near(b) / near(a)))] for a, b in ((1000, 10000), (10000, 100000))]
gt = []
for n in (10**4, 10**5, 10**6, 10**7, len(tr)):
    cc = collections.Counter(tr[:n]); N1 = sum(1 for v in cc.values() if v == 1)
    unseen = sum(v for w, v in hc.items() if w not in cc) / Hn
    gt.append(dict(n=n, types=len(cc), gt=N1 / n, actual=unseen))
# random-order control: shuffle training words, GT vs actual on a random held-out sample of the same pool
rng = np.random.default_rng(0); pool = np.array(tr[:2_000_000]); rng.shuffle(pool)
ctrl = []
for n in (10**4, 10**5, 10**6):
    cc = collections.Counter(pool[:n].tolist()); N1 = sum(1 for v in cc.values() if v == 1); test = pool[1_500_000:1_600_000]
    ctrl.append(dict(n=n, gt=N1 / n, actual=float(np.mean([w not in cc for w in test]))))
out = dict(heaps=[[int(a), int(b)] for a, b in zip(ns, D)], beta_all=beta_all, beta_hi=beta_hi, alpha=alpha, resid=res, slopes=slopes, gt=gt, gt_shuffled=ctrl,
           n_words=len(tr), n_types=len(c), ho_words=Hn)
json.dump(out, open("ledger.json", "w"))
print({k: v for k, v in out.items() if k not in ("heaps", "resid")})
