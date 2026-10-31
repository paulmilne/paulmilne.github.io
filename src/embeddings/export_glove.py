"""Export the 20,000 most frequent GloVe words, unit-normalised and quantised to 8 bits, for Part 1's explorer.

    python3 export_glove.py glove100k.npy glove100k_words.json out.json   (inputs written by glove_an.py)
"""
import base64, json, sys
import numpy as np
E = np.load(sys.argv[1]); W = json.load(open(sys.argv[2])); N = 20000
U = E[:N] / np.linalg.norm(E[:N], axis=1, keepdims=True)
s = np.abs(U).max(1, keepdims=True)
q = np.round(U / s * 127).astype(np.int8)
json.dump(dict(words=W[:N], dim=100, scale=[round(float(x), 6) for x in s[:, 0]], q=base64.b64encode(q.tobytes()).decode()),
          open(sys.argv[3], "w"))
