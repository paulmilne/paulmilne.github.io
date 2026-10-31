"""GPT-2 small forward pass in NumPy, row-vector convention (x @ W + b), returning the internals.

Weights: wte/wpe and the twelve blocks, as published in openai-community/gpt2 (safetensors).
"""
import numpy as np
from safetensors.numpy import load_file


class GPT2:
    def __init__(self, *files, n_head=12, eps=1e-5):
        self.w = {}
        for f in files:
            self.w.update({k.removeprefix("transformer."): v.astype(np.float32) for k, v in load_file(f).items()})
        self.H = n_head; self.eps = eps
        self.L = 1 + max(int(k.split(".")[1]) for k in self.w if k.startswith("h."))

    def ln(self, x, p):
        m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + self.eps) * self.w[p + ".weight"] + self.w[p + ".bias"]

    @staticmethod
    def gelu(x):  # GPT-2 uses the tanh approximation
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, ids, keep=True):
        w = self.w; T = len(ids)
        x = w["wte.weight"][ids] + w["wpe.weight"][:T]                  # (T, d)
        d = x.shape[1]; H = self.H; dh = d // H
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)  # causal
        trace = dict(resid=[x.copy()], attn=[], attn_out=[], mlp_out=[])
        for l in range(self.L):
            p = f"h.{l}."
            a = self.ln(x, p + "ln_1")
            qkv = a @ w[p + "attn.c_attn.weight"] + w[p + "attn.c_attn.bias"]   # (T, 3d)
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(T, H, dh).transpose(1, 0, 2)                  # (H, T, dh)
            k = k.reshape(T, H, dh).transpose(1, 0, 2)
            v = v.reshape(T, H, dh).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(dh) + mask           # (H, T, T)
            s = np.exp(s - s.max(-1, keepdims=True)); A = s / s.sum(-1, keepdims=True)
            z = (A @ v).transpose(1, 0, 2).reshape(T, d)                # (T, d)
            o = z @ w[p + "attn.c_proj.weight"] + w[p + "attn.c_proj.bias"]
            x = x + o
            m = self.ln(x, p + "ln_2")
            f = self.gelu(m @ w[p + "mlp.c_fc.weight"] + w[p + "mlp.c_fc.bias"]) @ w[p + "mlp.c_proj.weight"] + w[p + "mlp.c_proj.bias"]
            x = x + f
            if keep:
                trace["attn"].append(A); trace["attn_out"].append(o); trace["mlp_out"].append(f); trace["resid"].append(x.copy())
        logits = self.ln(x, "ln_f") @ w["wte.weight"].T                 # tied output matrix
        return logits, trace

    def lens(self, x):
        """Logit lens: apply the final LayerNorm and the unembedding to an intermediate residual."""
        return self.ln(x, "ln_f") @ self.w["wte.weight"].T
