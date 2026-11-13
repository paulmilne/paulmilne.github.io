from mu import bfs, construct
from itertools import product
import json, time
t=time.time()
for cap in (14, 18, 22):
    depth, parent = bfs(10**9, max_len=cap)
    res = {}
    for L in range(1, 9):
        for tup in product("IU", repeat=L):
            x="".join(tup)
            if x.count("I")%3: res[x]=depth.get(x)
    miss=[k for k,v in res.items() if v is None]
    print(cap, len(depth), "missing", len(miss), "max", max(v for v in res.values() if v is not None), round(time.time()-t,1))
json.dump({k:[v,len(construct(k))] for k,v in res.items()}, open("shortest.json","w"))
hard=sorted(res.items(), key=lambda kv:-(kv[1] or 99))[:8]; print(hard)
