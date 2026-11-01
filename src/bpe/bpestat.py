"""BPE on text8 with three merge rules, tracking compression and code-length statistics.

  python3 bpestat.py RULE N_MERGES TRAIN_CHARS OUT.json
RULE: freq (classic BPE: largest pair count), pmi (largest n_ab/(n_a n_b), WordPiece-style),
      gain (largest exact decrease in unigram code length).
Held-out = last 5M characters of text8, segmented incrementally with the same merges.
"""
import collections, json, math, sys, time
import numpy as np
EOW = "_"
rule, NM, TRAIN, out = sys.argv[1], int(sys.argv[2]), int(float(sys.argv[3])), sys.argv[4]
text = open("/home/claude/emb/dl/text8").read()
tr = text[:TRAIN].split(); ho = text[95_000_000:].split()
if TRAIN < 95_000_000: tr = tr[:-1]          # drop the possibly cut last word
ctr, cho = collections.Counter(tr), collections.Counter(ho)
types = sorted(set(ctr) | set(cho))
syms = [list(w) + [EOW] for w in types]
ftr = [ctr.get(w, 0) for w in types]; fho = [cho.get(w, 0) for w in types]
Ctr = sum((len(w) + 1) * c for w, c in ctr.items()); Cho = sum((len(w) + 1) * c for w, c in cho.items())
pairs = collections.Counter(); where = collections.defaultdict(set)
symc = collections.Counter(); symh = collections.Counter()
for i, s in enumerate(syms):
    for t in s: symc[t] += ftr[i]; symh[t] += fho[i]
    for p in zip(s, s[1:]):
        if ftr[i]: pairs[p] += ftr[i]
        where[p].add(i)
for k in [k for k, v in symc.items() if v == 0]: del symc[k]
V0 = 27
def stats(k):
    n = np.array([v for v in symc.values() if v > 0], float); T = n.sum(); p = n / T
    H = -(p * np.log2(p)).sum(); a = 2.5; Ha = math.log2((p ** a).sum()) / (1 - a)
    V = V0 + k
    f = np.sort(n)[::-1]; r = np.arange(1, len(f) + 1); m = (r >= 10) & (r <= min(len(f), 2000))
    zipf = float(-np.polyfit(np.log(r[m]), np.log(f[m]), 1)[0]) if m.sum() > 10 else None
    Th = sum(symh.values()); den = T + 0.5 * V
    xent = -sum(c * math.log2((symc.get(t, 0) + 0.5) / den) for t, c in symh.items() if c > 0)
    single = sum(fho[i] for i, s in enumerate(syms) if len(s) == 1) / sum(fho)
    return dict(k=k, V=V, used=len(n), T=int(T), cpt=Ctr / T, H=float(H), bpc=float(H * T / Ctr), renyi=float(Ha / math.log2(V)),
                shannon_eff=float(H / math.log2(V)), zipf=zipf, Th=int(Th), cpt_ho=Cho / Th, bpc_ho=xent / Cho, single_ho=single)
cps = sorted(set([0] + [int(round(2 ** (x / 4))) for x in range(0, 4 * int(math.log2(NM)) + 1)] + [NM]))
log = [stats(0)]; merges = []; t0 = time.time()
for k in range(NM):
    if not pairs: break
    if rule == "freq":
        (a, b), c = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
    else:
        keys = list(pairs.keys()); nab = np.fromiter((pairs[q] for q in keys), float, len(keys))
        na = np.fromiter((symc[q[0]] for q in keys), float, len(keys)); nb = np.fromiter((symc[q[1]] for q in keys), float, len(keys))
        N = float(sum(symc.values()))
        if rule == "pmi":
            sc = nab / (na * nb); sc[nab < 2] = -1
        else:  # exact change in unigram code length (nats), with x log x terms; overlapping pairs (a==b) approximated
            xl = lambda x: np.where(x > 0, x * np.log(np.maximum(x, 1e-300)), 0.0)
            same = np.array([q[0] == q[1] for q in keys])
            na2 = np.where(same, na - 2 * nab, na - nab); nb2 = np.where(same, 0.0, nb - nab)
            dL = (xl(N - nab) - xl(N)) - (xl(na2) - xl(na) + np.where(same, 0.0, xl(nb2) - xl(nb)) + xl(nab))
            sc = -dL; sc[nab < 2] = -1e300
        j = int(np.argmax(sc)); (a, b), c = keys[j], pairs[keys[j]]
        if sc[j] <= -1: break
    if c < 2: break
    ab = a + b; nm_tr = 0
    for i in list(where[(a, b)]):
        s = syms[i]; f = ftr[i]; g = fho[i]
        for p in zip(s, s[1:]):
            if f:
                pairs[p] -= f
                if pairs[p] <= 0: del pairs[p]
            where[p].discard(i)
        j = 0; o = []
        while j < len(s):
            if j < len(s) - 1 and s[j] == a and s[j + 1] == b:
                o.append(ab); j += 2; nm_tr += f
                symc[a] -= f; symc[b] -= f; symc[ab] += f; symh[a] -= g; symh[b] -= g; symh[ab] += g
            else: o.append(s[j]); j += 1
        syms[i] = o
        for p in zip(o, o[1:]):
            if f: pairs[p] += f
            where[p].add(i)
    where.pop((a, b), None)
    for q in (a, b):
        if symc[q] <= 0: del symc[q]
    merges.append([a, b, int(c)])
    if k + 1 in cps:
        st = stats(k + 1); log.append(st)
        print(rule, TRAIN, k + 1, a, b, c, "cpt %.3f bpc %.4f bpc_ho %.4f" % (st["cpt"], st["bpc"], st["bpc_ho"]), "%.0fs" % (time.time() - t0), flush=True)
if log[-1]["k"] != len(merges): log.append(stats(len(merges)))
final = collections.Counter()
for i, s in enumerate(syms):
    for t in s: final[t] += ftr[i]
json.dump(dict(rule=rule, train_chars=TRAIN, Ctr=Ctr, Cho=Cho, log=log, merges=merges,
               top=final.most_common(3000)), open(out, "w"))
print("done", time.time() - t0)
