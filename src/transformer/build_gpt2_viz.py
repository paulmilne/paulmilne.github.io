"""Part 2 data: GPT-2 small internals on example texts (attention, residual norms, logit lens, head scores)."""
import json, sys, base64, subprocess
import numpy as np
from gpt2_np import GPT2

g = GPT2('models/gpt2-embeddings.safetensors', 'models/gpt2-blocks-0-5.safetensors', 'models/gpt2-blocks-6-11.safetensors')
V = json.load(open('models/gpt2-vocab.json'))
texts = [
    ("Indirect object", "When Mary and John went to the shops, John gave a drink to"),
    ("Copying a name", "Bill Tindall wrote memos about the lunar module. Everyone at the space center read the memos of Bill"),
    ("A fact", "The Eiffel Tower is in the city of"),
    ("Sentence", "The cat sat on the mat because it was warm, and the dog lay by the fire because it was tired."),
]
ids_list = json.loads(subprocess.run(["node", "enc.js"], input=json.dumps([t for _, t in texts]), capture_output=True, text=True).stdout)
rng = np.random.default_rng(3)
rand = list(rng.integers(1000, 20000, 25))
seqs = [(n, t, [50256] + ids) for (n, t), ids in zip(texts, ids_list)]
seqs.append(("Repeated random tokens", None, [50256] + rand + rand))

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

out = []
for name, text, ids in seqs:
    lg, tr = g.forward(np.array(ids))
    T = len(ids)
    A = np.stack(tr["attn"])                      # (L, H, T, T)
    q = np.round(A * 255).astype(np.uint8)
    final_top = lg.argmax(-1)
    lens = []
    for l, r in enumerate(tr["resid"]):          # 13 residual states: after embedding, after each block
        P = softmax(g.lens(r))
        top = P.argmax(-1)
        lens.append([[V[int(top[t])], round(float(P[t, top[t]]), 3), round(float(P[t, final_top[t]]), 3)] for t in range(T)])
    norms = [[round(float(x), 2) for x in np.linalg.norm(r, axis=-1)] for r in tr["resid"]]
    contrib = [[round(float(np.linalg.norm(a[t])), 2) for t in range(T)] for a in tr["attn_out"]]
    mcontrib = [[round(float(np.linalg.norm(m[t])), 2) for t in range(T)] for m in tr["mlp_out"]]
    Pf = softmax(lg[-1]); top5 = np.argsort(-Pf)[:5]
    out.append(dict(name=name, tokens=[V[i] for i in ids], T=T, attn=base64.b64encode(q.tobytes()).decode(),
                    lens=lens, norms=norms, attn_norm=contrib, mlp_norm=mcontrib,
                    next=[[V[int(i)], round(float(Pf[i]), 4)] for i in top5]))
    print(name, T, [V[int(i)] for i in top5[:3]])

# head scores from the random repeated sequence
_, tr = g.forward(np.array(seqs[-1][2])); A = np.stack(tr["attn"]); n = len(rand)
pos = np.arange(n + 1, 2 * n + 1)                # second copy (index 0 is <|endoftext|>)
induction = A[:, :, pos, pos - n + 1].mean(-1)   # attend to the token after the earlier copy
prev = np.stack([A[:, :, t, t - 1] for t in range(2, 2 * n + 1)], -1).mean(-1)
sink = A[:, :, 1:, 0].mean(-1)
scores = dict(induction=np.round(induction, 3).tolist(), previous=np.round(prev, 3).tolist(), first=np.round(sink, 3).tolist())
top_ind = sorted([(float(induction[l, h]), l, h) for l in range(12) for h in range(12)], reverse=True)[:6]
top_prev = sorted([(float(prev[l, h]), l, h) for l in range(12) for h in range(12)], reverse=True)[:6]
print("induction", top_ind); print("previous", top_prev); print("mean first-token attention", float(sink.mean()))
json.dump(dict(examples=out, scores=scores, n_rand=n), open(sys.argv[1], "w"), separators=(",", ":"), ensure_ascii=False)
