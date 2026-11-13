"""MIU system: exhaustive search, the I-count invariant, and a constructive
derivation for every string that the invariant allows."""
from collections import deque
from itertools import product

def successors(s):
    """All (rule, position, result) one step from s (s excludes the leading M)."""
    out = []
    if s.endswith("I"):
        out.append((1, len(s), s + "U"))
    out.append((2, 0, s + s))
    for i in range(len(s) - 2):
        if s[i:i+3] == "III":
            out.append((3, i, s[:i] + "U" + s[i+3:]))
    for i in range(len(s) - 1):
        if s[i:i+2] == "UU":
            out.append((4, i, s[:i] + s[i+2:]))
    return out

def bfs(max_steps, max_len=None):
    """Theorems by number of steps from MI, with parent pointers."""
    depth, parent = {"I": 0}, {"I": None}
    q = deque(["I"])
    while q:
        s = q.popleft()
        if depth[s] == max_steps:
            continue
        for r, i, t in successors(s):
            if max_len and len(t) > max_len:
                continue
            if t not in depth:
                depth[t], parent[t] = depth[s] + 1, (s, r, i)
                q.append(t)
    return depth, parent

def construct(x):
    """A derivation of Mx from MI when #I(x) is not a multiple of 3, else None.
    Returns a list of (rule, position, string)."""
    c, u = x.count("I"), x.count("U")
    if c % 3 == 0:
        return None
    steps, s = [], "I"
    while not (len(s) >= c + 3 * u and len(s) % 3 == c % 3):
        s = s + s; steps.append((2, 0, s))
    r = (len(s) - c - 3 * u) // 3          # spare triples -> trailing U's
    if r % 2:                              # make the trailing U's pair up
        s = s + "U"; steps.append((1, len(s) - 1, s)); r += 1
    pos = 0                                # build x left to right
    for ch in x:
        if ch == "U":
            s = s[:pos] + "U" + s[pos+3:]; steps.append((3, pos, s))
        pos += 1
    while s[pos:pos+3] == "III":           # remaining I's -> U's
        s = s[:pos] + "U" + s[pos+3:]; steps.append((3, pos, s)); pos += 1
    while s.endswith("UU") and len(s) > len(x):
        s = s[:-2]; steps.append((4, len(s), s))
    assert s == x, (x, s)
    return steps

def check(steps, start="I"):
    s = start
    for r, i, t in steps:
        assert any(t == t2 for r2, i2, t2 in successors(s) if r2 == r), (s, r, t)
        s = t
    return s

if __name__ == "__main__":
    # constructive solver: every valid string up to length 10
    n = 0
    for L in range(1, 11):
        for tup in product("IU", repeat=L):
            x = "".join(tup)
            d = construct(x)
            if x.count("I") % 3:
                assert d is not None and check(d) == x; n += 1
            else:
                assert d is None
    print("constructive derivations verified for", n, "strings up to length 10")
    depth, _ = bfs(6)
    print("theorems within 6 steps:", len(depth), " any with #I % 3 == 0:",
          any(s.count("I") % 3 == 0 for s in depth))
