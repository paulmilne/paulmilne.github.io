import numpy as np, json, sys
from ff import *
from multiprocessing import Pool
def tsync(rs,th=0.9):
    t,r=rs[:,0],rs[:,1]; bad=np.where(r<th)[0]
    if len(bad)==0: return 0.0
    if bad[-1]==len(r)-1: return None
    return float(t[bad[-1]+1])
xy=np.array([(i%5,i//5) for i in range(20)],float)
D=np.hypot(*(xy[:,None,:]-xy[None,:,:]).transpose(2,0,1))
REAL=dict(eps=0.02,spread=0.02,delay=0.003,loss=0.1,refr=0.05,jitter=0.002)
def topo(args):
    rad,s=args; adj=(D<rad)&(D>0) if rad else None
    fl,rs=sim(adj=adj,Tend=180,seed=s,**REAL); return rad,s,tsync(rs)
def grid(args):
    eps,sp,s=args; fl,rs=sim(eps=eps,spread=sp,delay=0.003,loss=0.1,refr=0.05,jitter=0.002,Tend=120,seed=s)
    r=rs[rs[:,0]>90,1]; return eps,sp,s,float(r.mean()),float(r.min())
if __name__=='__main__':
    with Pool(8) as P:
        T=P.map(topo,[(rad,s) for rad in (1.01,1.5,2.3,0) for s in range(12)])
        json.dump(T,open('topo.json','w'))
        E=np.round(np.geomspace(0.003,0.2,10),4); S=np.round(np.linspace(0,0.1,11),3)
        G=P.map(grid,[(e,sp,s) for e in E for sp in S for s in range(3)])
        json.dump(G,open('grid.json','w'))
    import collections
    for rad in (1.01,1.5,2.3,0):
        v=[x[2] for x in T if x[0]==rad]; print(rad,np.round([x if x is not None else -1 for x in v],1))
