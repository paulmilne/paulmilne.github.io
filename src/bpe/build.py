import json, collections, math
R = {}
keys = dict(freq="freq_full", gain="gain_full", pmi="pmi_full", p1e5="freq_1e5", p1e6="freq_1e6", p1e7="freq_1e7")
fields = ["k", "V", "T", "cpt", "cpt_ho", "bpc", "bpc_ho", "renyi", "shannon_eff", "zipf", "single_ho", "H"]
raw = {}
for k, f in keys.items():
    d = json.load(open(f"runs/{f}.json")); raw[k] = d
    R[k] = [{x: (round(r[x], 5) if isinstance(r[x], float) else r[x]) for x in fields} for r in d["log"]]
first = {k: raw[k]["merges"][:40] for k in ("freq", "gain", "pmi")}
def toks(merges, n): 
    out = []
    for a, b, c in merges[:n]: out.append(a + b)
    return out
overlap = []
for n in (100, 500, 1000, 2000):
    a, b = set(toks(raw["freq"]["merges"], n)), set(toks(raw["gain"]["merges"], n)); overlap.append(dict(n=n, frac=len(a & b) / n))
# Zipf curves: re-segment training word counts with first K merges (K<=2000) + final top list
text = open("/home/claude/emb/dl/text8").read()[:95_000_000].split()[:-1]
cnt = collections.Counter(text)
def seg_counts(merges):
    rank = {(a, b): i for i, (a, b, c) in enumerate(merges)}
    tc = collections.Counter()
    for w, c in cnt.items():
        s = list(w) + ["_"]
        while len(s) > 1:
            best = min(((rank.get((s[i], s[i + 1]), 1e18), i) for i in range(len(s) - 1)))
            if best[0] == 1e18: break
            i = best[1]; s = s[:i] + [s[i] + s[i + 1]] + s[i + 2:]
        for t in s: tc[t] += c
    return tc
def curve(tc, T=None):
    f = sorted(tc.values(), reverse=True); T = T or sum(f); out = []; r = 1
    while r <= len(f):
        out.append([r, f[r - 1] / T]); r = max(r + 1, int(r * 1.15))
    if out[-1][0] != len(f): out.append([len(f), f[-1] / T])
    return out
zipf = []
for K in (0, 30, 300, 2000, len(raw["freq"]["merges"])):
    zipf.append(dict(k=K, pts=curve(seg_counts(raw["freq"]["merges"][:K]))))
gpt = json.load(open("gpt_counts.json"))
newest = {k: [[i + 1, m[2]] for i, m in enumerate(raw[k]["merges"]) if i < 20 or (i + 1) in set(int(1.08 ** j) for j in range(200)) or i == len(raw[k]["merges"]) - 1] for k in ("p1e5", "p1e6", "p1e7", "freq")}
Ctr = {k: raw[k]["Ctr"] for k in raw}
json.dump(dict(runs=R, newest=newest, Ctr=Ctr, first=first, overlap=overlap, zipf=zipf, gpt=gpt), open("post/data/bpe.json", "w"), separators=(",", ":"))
print(overlap); print({k: (v[-1]["k"], round(min(r["bpc_ho"] for r in v), 4), min(v, key=lambda r: r["bpc_ho"])["k"]) for k, v in R.items()})
