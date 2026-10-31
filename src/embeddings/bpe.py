"""Byte-pair encoding (Sennrich et al. 2016 style) on text8 word counts, with incremental pair counts."""
import collections, json, time, heapq
EOW = "_"
def learn(counts, n_merges):
    words = {tuple(w) + (EOW,): c for w, c in counts.items()}
    vocab = list(words.items())
    syms = [list(w) for w, _ in vocab]; freq = [c for _, c in vocab]
    pairs = collections.Counter(); where = collections.defaultdict(set)
    for i, s in enumerate(syms):
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += freq[i]; where[(a, b)].add(i)
    merges = []; snapshots = {}
    for k in range(n_merges):
        if not pairs: break
        (a, b), c = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if c < 2: break
        merges.append([a, b, c]); ab = a + b
        for i in list(where[(a, b)]):
            s = syms[i]; f = freq[i]
            # remove old pairs
            for p in zip(s, s[1:]):
                pairs[p] -= f
                if pairs[p] <= 0: del pairs[p]
                where[p].discard(i)
            j = 0; out = []
            while j < len(s):
                if j < len(s) - 1 and s[j] == a and s[j + 1] == b: out.append(ab); j += 2
                else: out.append(s[j]); j += 1
            syms[i] = out
            for p in zip(out, out[1:]):
                pairs[p] += f; where[p].add(i)
        where.pop((a, b), None)
        if k + 1 in (10, 100, 1000, 4000, 8000, 16000):
            ntok = sum(len(s) * f for s, f in zip(syms, freq)); nw = sum(freq)
            snapshots[k + 1] = dict(tokens_per_word=round(ntok / nw, 3))
            print(k + 1, a, b, c, snapshots[k + 1], flush=True)
    return merges, snapshots
if __name__ == "__main__":
    t = time.time()
    words = open("dl/text8").read().split()
    counts = collections.Counter(words)
    print("words", len(words), "types", len(counts))
    merges, snaps = learn(counts, 16000)
    base = sum(len(w) + 1 for w in words) / len(words)
    json.dump(dict(merges=merges, snapshots=snaps, chars_per_word=round(base, 3), n_words=len(words), n_types=len(counts)),
              open("bpe_text8.json", "w"))
    print("time", time.time() - t)
