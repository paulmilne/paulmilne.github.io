"""GPT-2 embedding-table statistics for Part 1.

    python3 gpt2_stats.py gpt2-embeddings.safetensors gpt2-vocab.json out.json
"""
import json, sys
import numpy as np
from safetensors.numpy import load_file

st = load_file(sys.argv[1])
wte = next(v for k, v in st.items() if k.endswith("wte.weight")).astype(np.float64)
wpe = next(v for k, v in st.items() if k.endswith("wpe.weight")).astype(np.float64)
vocab = json.load(open(sys.argv[2]))                  # id -> text, from the r50k_base (GPT-2) encoding
def text(i):
    return vocab[i]

V, d = wte.shape
norms = np.linalg.norm(wte, axis=1)
U = wte / norms[:, None]
mu = U.mean(0); mdir = mu / np.linalg.norm(mu)
cos_to_mean = U @ mdir
rng = np.random.default_rng(0)
a = rng.integers(0, V, 200000); b = rng.integers(0, V, 200000); keep = a != b
mean_cos = float((U[a[keep]] * U[b[keep]]).sum(1).mean())

ids = np.arange(V)
samp = rng.choice(V, 4000, replace=False)
bins = np.linspace(0, V, 41)
binned = [[float((bins[i] + bins[i + 1]) / 2), float(np.median(norms[(ids >= bins[i]) & (ids < bins[i + 1])]))] for i in range(40)]

# tokens nearest the mean direction (highest cosine), and the ten with the smallest norms
centroid = [text(int(i)) for i in np.argsort(-cos_to_mean)[:200]
            if text(int(i)) and text(int(i)).isprintable() and '\ufffd' not in text(int(i))][:20]
band = lambda a, b: float(np.median(norms[a:b]))
norm_bands = dict(first1000=band(0, 1000), last20000=band(30000, V))
smallest = [text(int(i)) for i in np.argsort(norms)[:20]]

# neighbours for whole-word tokens with a leading space among the first 20,000 ids, plus some without
cand = [i for i in range(V) if vocab[i].startswith(" ") and vocab[i][1:].isalpha() and i < 20000][:6000]
cand += [i for i in range(V) if vocab[i].isalpha() and i < 3000][:1500]
nn = {}
for start in range(0, len(cand), 1000):
    blk = cand[start:start + 1000]
    S = U[blk] @ U.T
    for row, i in zip(S, blk):
        o = np.argpartition(-row, 9)[:9]; o = o[np.argsort(-row[o])]
        nn[text(i)] = [[text(int(j)), round(float(row[j]), 3)] for j in o if j != i][:8]

# position embeddings: first three principal components
P = wpe - wpe.mean(0)
Uu, S, Vt = np.linalg.svd(P, full_matrices=False)
pcs = (P @ Vt[:3].T)
var3 = float((S[:3] ** 2).sum() / (S ** 2).sum())

out = dict(V=V, d=d, norm_min=float(norms.min()), norm_max=float(norms.max()), norm_median=float(np.median(norms)),
           mean_cos=mean_cos, mean_dir_angle=float(np.degrees(np.arccos(cos_to_mean.mean()))),
           norm_sample=[[int(i), round(float(norms[i]), 3)] for i in samp], norm_binned=binned,
           centroid_tokens=centroid, norm_bands=norm_bands, smallest_norm_tokens=smallest, nn=nn,
           wpe_pcs=[[round(float(x), 3) for x in r] for r in pcs], wpe_var3=var3,
           tied=bool("lm_head.weight" not in st))
json.dump(out, open(sys.argv[3], "w"), separators=(",", ":"), ensure_ascii=False)
print({k: v for k, v in out.items() if k not in ("norm_sample", "norm_binned", "nn", "wpe_pcs")})
print("king", nn.get(" king")); print("bank", nn.get(" bank"))
