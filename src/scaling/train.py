"""Small character-level GPTs on text8 for a scaling-law study.

    python3 train.py --d 64 --L 4 --tokens 2e6 --out runs/x.json
Logs training loss and held-out validation loss (nats per character) along the run.
"""
import argparse, json, math, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--d", type=int, required=True); p.add_argument("--L", type=int, required=True)
p.add_argument("--tokens", type=float, required=True); p.add_argument("--ctx", type=int, default=128)
p.add_argument("--batch", type=int, default=32); p.add_argument("--lr", type=float, default=None)
p.add_argument("--seed", type=int, default=0); p.add_argument("--out", required=True)
p.add_argument("--threads", type=int, default=1)
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)

raw = np.fromfile("data/text8.u8", dtype=np.uint8)            # 0..26
train, val = raw[:95_000_000], raw[95_000_000:]
V = 27

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.ln2 = nn.LayerNorm(d); s.f1 = nn.Linear(d, 4 * d); s.f2 = nn.Linear(4 * d, d)
    def forward(s, x):
        B, T, d = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(d, -1)
        q, k, v = (t.view(B, T, s.h, d // s.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + s.o(y)
        return x + s.f2(F.gelu(s.f1(s.ln2(x))))

class GPT(nn.Module):
    def __init__(s, d, L, ctx):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList([Block(d, max(1, d // 32)) for _ in range(L)])
        s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, V, bias=False)
    def forward(s, idx):
        x = s.emb(idx) + s.pos(torch.arange(idx.shape[1]))
        for b in s.blocks: x = b(x)
        return s.head(s.lnf(x))

m = GPT(a.d, a.L, a.ctx)
N = sum(p.numel() for n, p in m.named_parameters() if not (n.startswith("emb") or n.startswith("pos") or n.startswith("head")))
N_all = sum(p.numel() for p in m.parameters())
tok_per_step = a.batch * a.ctx
steps = max(50, int(a.tokens / tok_per_step))
lr = a.lr or min(3e-3, 0.003 * (128 / a.d) ** 0.5)
opt = torch.optim.AdamW(m.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
warm = max(10, steps // 50)
sched = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))

rng = np.random.default_rng(a.seed)
def batch(src, n):
    ix = rng.integers(0, len(src) - a.ctx - 1, n)
    x = np.stack([src[i:i + a.ctx] for i in ix]); y = np.stack([src[i + 1:i + a.ctx + 1] for i in ix])
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

vrng = np.random.default_rng(1234)
vix = vrng.integers(0, len(val) - a.ctx - 1, 256)
VX = torch.from_numpy(np.stack([val[i:i + a.ctx] for i in vix]).astype(np.int64))
VY = torch.from_numpy(np.stack([val[i + 1:i + a.ctx + 1] for i in vix]).astype(np.int64))
def evaluate():
    m.eval()
    with torch.no_grad():
        tot = 0.0
        for i in range(0, 256, 64):
            lg = m(VX[i:i + 64]); tot += F.cross_entropy(lg.reshape(-1, V), VY[i:i + 64].reshape(-1), reduction="sum").item()
    m.train(); return tot / VY.numel()

log = []; t0 = time.time(); ema = None
n_eval = 12; eval_at = set(int(round(steps * (k / n_eval) ** 1.5)) for k in range(1, n_eval + 1))
for s in range(steps):
    for g in opt.param_groups: g["lr"] = lr * sched(s)
    x, y = batch(train, a.batch)
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if (s + 1) in eval_at or s == steps - 1:
        log.append(dict(step=s + 1, tokens=(s + 1) * tok_per_step, train=round(ema, 4), val=round(evaluate(), 4), sec=round(time.time() - t0, 1)))
json.dump(dict(d=a.d, L=a.L, N=N, N_all=N_all, tokens=steps * tok_per_step, ctx=a.ctx, batch=a.batch, lr=lr, steps=steps,
               flops=6 * N * steps * tok_per_step, flops_attn=6 * a.L * a.ctx * a.d * steps * tok_per_step,
               final_val=log[-1]["val"], seconds=round(time.time() - t0, 1), log=log), open(a.out, "w"))
print(f"d={a.d} L={a.L} N={N} D={steps*tok_per_step:.3g} val={log[-1]['val']:.4f} {time.time()-t0:.0f}s")
