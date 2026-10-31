"""Data for 'Embeddings, Part 2': the blog's own passages embedded with all-MiniLM-L6-v2 (numpy forward pass),
BM25 / dense / hybrid retrieval on hand-written test queries, quantisation and truncation, uniformity, a t-SNE map.

    python3 build_part2.py MODEL_DIR blog_chunks.json queries.json out.json
"""
import base64, json, sys, time
import numpy as np
from minilm_np import MiniLM
from bm25 import BM25

mdir, chunks_path, q_path, out_path = sys.argv[1:5]
m = MiniLM(f"{mdir}/minilm.safetensors", f"{mdir}/minilm-tokenizer.json", f"{mdir}/minilm-config.json")
B = json.load(open(chunks_path)); Q = json.load(open(q_path))
posts, chunks = B["posts"], B["chunks"]
docs = [posts[c["post"]]["title"] + ". " + c["heading"] + ". " + c["text"] for c in chunks]

t = time.time()
X = m.embed_many(docs)
ntok = [len(m.tok.encode(d).ids) for d in docs]
print("embedded", X.shape, round(time.time() - t, 1), "s; truncated at 256:", sum(n >= 256 for n in ntok))
QV = m.embed_many([q for q, _ in Q])
cpost = np.array([c["post"] for c in chunks])


def post_rank(scores):
    best = {}
    for s, p in zip(scores, cpost):
        best[p] = max(best.get(p, -1e9), s)
    return sorted(best, key=lambda p: -best[p])


def rrf(*rankings, k=60):
    sc = {}
    for r in rankings:
        for i, p in enumerate(r):
            sc[p] = sc.get(p, 0) + 1 / (k + i + 1)
    return sorted(sc, key=lambda p: -sc[p])


bm = BM25(docs)
def evaluate(rank_fn):
    firsts = []
    for qi, (q, rel) in enumerate(Q):
        r = rank_fn(qi, q)
        firsts.append(min(r.index(p) + 1 for p in rel if p in r))
    f = np.array(firsts)
    return dict(hit1=float((f <= 1).mean()), hit3=float((f <= 3).mean()), mrr=float((1 / f).mean()), firsts=f.tolist())

dense_rank = lambda qi, q: post_rank(X @ QV[qi])
bm_rank = lambda qi, q: post_rank(bm.scores(q))
ev = dict(bm25=evaluate(bm_rank), dense=evaluate(dense_rank),
          hybrid=evaluate(lambda qi, q: rrf(bm_rank(qi, q), dense_rank(qi, q))))
for k, v in ev.items():
    print(k, {a: round(b, 3) for a, b in v.items() if a != "firsts"})

# exact top-10 neighbours of each chunk, then recall under compression
def topk(S, k=10):
    S = S.copy(); np.fill_diagonal(S, -np.inf); return np.argsort(-S, 1)[:, :k]
ref = topk(X @ X.T)
def recall(nb): return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ref, nb)]))
sc8 = np.abs(X).max(1, keepdims=True); X8 = np.round(X / sc8 * 127) / 127 * sc8
Xb = np.sign(X); ham = Xb @ Xb.T
comp = [dict(name="float32 (exact)", bytes=384 * 4, recall=1.0, mrr=ev["dense"]["mrr"])]
comp.append(dict(name="int8", bytes=384, recall=recall(topk(X8 @ X8.T)),
                 mrr=evaluate(lambda qi, q: post_rank(X8 @ QV[qi]))["mrr"]))
comp.append(dict(name="1 bit (sign), Hamming", bytes=48, recall=recall(topk(ham)),
                 mrr=evaluate(lambda qi, q: post_rank(Xb @ np.sign(QV[qi])))["mrr"]))
# binary first pass for 50 candidates, rescored with float vectors
def bin_rescore(qi, q, n=50):
    cand = np.argsort(-(Xb @ np.sign(QV[qi])))[:n]
    s = np.full(len(X), -9.0); s[cand] = X[cand] @ QV[qi]; return post_rank(s)
comp.append(dict(name="1 bit, top 50 rescored", bytes=48, recall=None, mrr=evaluate(bin_rescore)["mrr"]))
for dim in (192, 96, 48):
    Xt = X[:, :dim] / np.linalg.norm(X[:, :dim], axis=1, keepdims=True)
    Qt = QV[:, :dim] / np.linalg.norm(QV[:, :dim], axis=1, keepdims=True)
    comp.append(dict(name=f"first {dim} dimensions", bytes=dim * 4, recall=recall(topk(Xt @ Xt.T)),
                     mrr=evaluate(lambda qi, q, Xt=Xt, Qt=Qt: post_rank(Xt @ Qt[qi]))["mrr"]))
for c in comp: print(c)

# geometry: uniformity, mean cosine, alignment of consecutive chunks of the same post
rng = np.random.default_rng(0)
i = rng.integers(0, len(X), 20000); j = rng.integers(0, len(X), 20000); ok = i != j
d2 = ((X[i[ok]] - X[j[ok]]) ** 2).sum(1)
uniform = float(np.log(np.mean(np.exp(-2 * d2))))
mean_cos = float((X[i[ok]] * X[j[ok]]).sum(1).mean())
same = [(a, a + 1) for a in range(len(X) - 1) if cpost[a] == cpost[a + 1]]
align = float(np.mean([((X[a] - X[b]) ** 2).sum() for a, b in same]))
cos_same = float(np.mean([X[a] @ X[b] for a, b in same]))
print("uniformity", uniform, "mean cos", mean_cos, "align(adjacent)", align, "cos adjacent", cos_same)

# t-SNE map
from sklearn.manifold import TSNE
Y = TSNE(n_components=2, perplexity=25, metric="cosine", init="pca", random_state=0).fit_transform(X)

q8 = np.round(X / sc8 * 127).astype(np.int8)
out = dict(model="sentence-transformers/all-MiniLM-L6-v2", dim=384, posts=posts,
           chunks=[dict(post=c["post"], heading=c["heading"], text=c["text"][:600], ntok=n) for c, n in zip(chunks, ntok)],
           emb=base64.b64encode(q8.tobytes()).decode(), scale=[round(float(s), 6) for s in sc8[:, 0]],
           queries=[dict(q=q, rel=rel, vec=[round(float(v), 5) for v in qv]) for (q, rel), qv in zip(Q, QV)],
           eval=ev, compression=comp,
           geometry=dict(uniformity=uniform, mean_cos=mean_cos, align_adjacent=align, cos_adjacent=cos_same, n_adjacent=len(same)),
           tsne=[[round(float(a), 2), round(float(b), 2)] for a, b in Y],
           bm25=dict(k1=1.2, b=0.75))
json.dump(out, open(out_path, "w"), separators=(",", ":"), ensure_ascii=False)
print("written", out_path)
