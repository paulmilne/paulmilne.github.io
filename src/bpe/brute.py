import itertools, random, collections
def apply(seq, a, b):
    o=[];i=0
    while i<len(seq):
        if i<len(seq)-1 and seq[i]==a and seq[i+1]==b: o.append(a+b); i+=2
        else: o.append(seq[i]); i+=1
    return o
def pairs(words):
    c=collections.Counter()
    for w in words:
        for p in zip(w,w[1:]): c[p]+=1
    return c
def best(words,k):
    if k==0: return sum(map(len,words)),[]
    bst=(sum(map(len,words)),[])
    for p in pairs(words):
        n,seq=best([apply(w,*p) for w in words],k-1)
        if n<bst[0]: bst=(n,[p]+seq)
    return bst
def greedy_best(words,k):  # best over all greedy tie-breaks
    if k==0: return sum(map(len,words)),[]
    c=pairs(words); 
    if not c: return sum(map(len,words)),[]
    m=max(c.values()); bst=None
    for p,v in c.items():
        if v==m:
            n,seq=greedy_best([apply(w,*p) for w in words],k-1)
            if bst is None or n<bst[0]: bst=(n,[p]+seq)
    return bst
random.seed(1); found=[]
for trial in range(20000):
    L=random.randint(5,12); s="".join(random.choice("abc") for _ in range(L))
    words=[list(s)]
    for k in (2,3):
        g=greedy_best(words,k); o=best(words,k)
        if o[0]<g[0]: found.append((len(s),s,k,g,o)); break
found.sort()
for f in found[:8]: print(f)
print("--- proper counts, words corpora")
def pairs(words):
    c=collections.Counter()
    for w in words:
        for p in set(zip(w,w[1:])):
            c[p]+=len(w)-len(apply(w,*p))
    return c
random.seed(2); found=[]
for trial in range(30000):
    nw=random.randint(1,4); words=[list("".join(random.choice("abcd") for _ in range(random.randint(2,6)))) for _ in range(nw)]
    for k in (2,3):
        g=greedy_best(words,k); o=best(words,k)
        if o[0]<g[0]: found.append((sum(map(len,words)),["".join(w) for w in words],k,g,o)); break
found.sort(key=lambda f:(f[0],len(f[1])))
for f in found[:10]: print(f)
