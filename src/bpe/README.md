# Statistics of BPE (post: ai/byte-pair-encoding-statistics)

- `bpestat.py RULE N_MERGES TRAIN_CHARS OUT.json` — BPE on text8 with merge rule `freq` (classic BPE), `pmi` (n_ab/(n_a n_b)) or `gain` (exact unigram code-length decrease). Logs compression, unigram code length (train and held-out), Rényi efficiency, Zipf slope.
- `jobs.txt` — the runs used (the freq runs were later rerun to save every merge).
- `build.py` — assembles `data/bpe.json` for the post from `runs/*.json`.
- `brute.py` — exhaustive search for corpora where greedy BPE is not optimal.
- `gptcount.js` — tokens per character of held-out text8 under GPT-2/GPT-4/GPT-4o encodings (needs the `gpt-tokenizer` npm package).

Data: `text8` (Matt Mahoney), first 95M characters for training, last 5M held out.
- `ledger.py` — word-level vocabulary as a variety ledger: Heaps curve, top-K residual, Good–Turing missing mass with a shuffled control.
