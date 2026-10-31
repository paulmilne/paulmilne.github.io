"""all-MiniLM-L6-v2 forward pass in numpy: BERT encoder, mean pooling, L2 normalisation.

Reads the Hugging Face safetensors weights and tokenizer.json; no deep-learning framework needed.
"""
import json
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer


class MiniLM:
    def __init__(self, weights, tokenizer, config):
        self.w = {k.removeprefix("bert."): v.astype(np.float32) for k, v in load_file(weights).items()}
        self.cfg = json.load(open(config))
        self.tok = Tokenizer.from_file(tokenizer)
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=256)
        self.L = self.cfg["num_hidden_layers"]; self.H = self.cfg["num_attention_heads"]
        self.eps = self.cfg.get("layer_norm_eps", 1e-12)

    def ln(self, x, p):
        m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + self.eps) * self.w[p + ".weight"] + self.w[p + ".bias"]

    def lin(self, x, p):
        return x @ self.w[p + ".weight"].T + self.w[p + ".bias"]

    @staticmethod
    def gelu(x):
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))

    def encode_ids(self, ids, return_tokens=False):
        w = self.w; n = len(ids)
        x = w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:n] \
            + w["embeddings.token_type_embeddings.weight"][0]
        x = self.ln(x, "embeddings.LayerNorm")
        d = x.shape[1]; hd = d // self.H
        for l in range(self.L):
            p = f"encoder.layer.{l}."
            q = self.lin(x, p + "attention.self.query").reshape(n, self.H, hd).transpose(1, 0, 2)
            k = self.lin(x, p + "attention.self.key").reshape(n, self.H, hd).transpose(1, 0, 2)
            v = self.lin(x, p + "attention.self.value").reshape(n, self.H, hd).transpose(1, 0, 2)
            s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
            s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(1, 0, 2).reshape(n, d)
            x = self.ln(x + self.lin(a, p + "attention.output.dense"), p + "attention.output.LayerNorm")
            h = self.gelu(self.lin(x, p + "intermediate.dense"))
            x = self.ln(x + self.lin(h, p + "output.dense"), p + "output.LayerNorm")
        return x

    def embed(self, text):
        e = self.tok.encode(text)
        x = self.encode_ids(np.array(e.ids))
        v = x.mean(0)                       # mean pooling (no padding in a single sequence)
        return v / np.linalg.norm(v), e.tokens

    def embed_many(self, texts):
        return np.vstack([self.embed(t)[0] for t in texts])
