"""A fixed-size model of the top 1,000 fortunes, driven by the empirical
distributions of growth, exit and entry from the Forbes lists.

Everything is measured relative to the wealth of the 1,000th person, so common
booms and slumps drop out and only relative movements matter.

Each year:  (1) every member draws a growth rate at random from the growth
rates of that year's actual top-1,000 members ("equal odds"), or from those in
their own rank group ("by rank");  (2) members leave the list at the observed
rate for their rank group;  (3) new members arrive, as many as actually joined
the top 1,000 that year, with wealth drawn from those actual newcomers;
(4) the top 1,000 are kept."""
import json, numpy as np, pandas as pd
EDGES=[0,10,100,1000]; LABS=['1–10','11–100','101–1000']
def load(path='data/'):
    rows=[]
    for y in range(2011,2027):
        for p in json.load(open(path+f'forbes/billionaires_{y}.json'))['personList']['personsLists']:
            rows.append(dict(year=y,name=p.get('personName'),worth=(p.get('finalWorth') or 0)/1000))
    f=pd.DataFrame(rows); f=f[f.worth>=1].drop_duplicates(['year','name'])
    return f.pivot(index='name',columns='year',values='worth')
def stats(x):
    """x: log wealth. Statistics of the top 1,000."""
    w=np.exp(np.sort(x)[::-1][:1000]); h=lambda k:(k-1)/np.log(w[:k-1]/w[k-1]).sum()
    return dict(t10=w[:10].sum()/w.sum()*100,t100=w[:100].sum()/w.sum()*100,a100=h(100),a1000=h(1000),t1=w[0]/w.sum()*100)
class Model:
    def __init__(self,W):
        self.W=W; self.years=list(range(2011,2026)); self.g={}; self.exit={}; self.new={}; self.start={}
        for y in self.years+[2026]:
            a=W[y].dropna().sort_values(ascending=False); self.start[y]=np.log(a.values[:1000])-np.log(a.values[999])
        for y in self.years:
            a=W[y].dropna().sort_values(ascending=False); b=W[y+1].dropna().sort_values(ascending=False)
            top=a.index[:1000]; top1=b.index[:1000]; dl=np.log(b.values[999])-np.log(a.values[999])
            rank=pd.Series(np.arange(1,1001),index=top); stay=[n for n in top if n in b.index]
            g=np.log(b[stay].values/a[stay].values)-dl; grp=np.digitize(rank[stay].values,EDGES[1:-1],right=True)
            self.g[y]=[g[grp==k] for k in range(3)]; self.g[y].append(g)
            gone=np.array([n not in b.index for n in top]); gr=np.digitize(rank.values,EDGES[1:-1],right=True)
            self.exit[y]=[gone[gr==k].mean() for k in range(3)]
            newc=[n for n in top1 if n not in top]; self.new[y]=np.log(b[newc].values)-np.log(b.values[999])
    def step(self,x,y,rng,by_rank):
        o=np.argsort(-x); rank=np.empty(len(x),int); rank[o]=np.arange(1,len(x)+1); grp=np.digitize(rank,EDGES[1:-1],right=True)
        g=np.empty(len(x)); keep=np.ones(len(x),bool)
        for k in range(3):
            m=grp==k
            g[m]=rng.choice(self.g[y][k] if by_rank else self.g[y][3],size=m.sum()); keep[m]=rng.random(m.sum())>=self.exit[y][k]
        x=(x+g)[keep]; x=np.concatenate([x,rng.choice(self.new[y],size=len(self.new[y]))])
        x=np.sort(x)[::-1][:1000]; return x-x[-1]
    def run(self,y0,years=None,horizon=None,pool=None,by_rank=False,nsim=1000,seed=0,shock=None,x0=None):
        rng=np.random.default_rng(seed); out=[]
        for s in range(nsim):
            x=(self.start[y0] if x0 is None else x0).copy()
            if shock: x=np.sort(x)[::-1]; x[:shock[0]]+=np.log(shock[1]); x=np.sort(x)[::-1]; x=x-x[-1]
            seq=years if years is not None else rng.choice(pool,size=horizon); path=[stats(x)]
            for y in seq: x=self.step(x,y,rng,by_rank); path.append(stats(x))
            out.append(path)
        return out
if __name__=='__main__':
    W=load(); M=Model(W)
    act={y:stats(M.start[y]) for y in range(2011,2027)}
    print('actual top-10 share:',{y:round(act[y]['t10'],1) for y in act}); print('actual a100:',{y:round(act[y]['a100'],2) for y in act})
    print('new members of the top 1000 per year',{y:len(M.new[y]) for y in M.years}); print('exit by rank group',{y:[round(e,3) for e in M.exit[y]] for y in (2012,2016,2020,2024)})
    print('sd of relative growth',{y:round(float(np.std(M.g[y][3])),3) for y in M.years})
    def rep(nm,o,y1):
        for k in ('t10','t100','a100'):
            v=np.array([p[-1][k] for p in o]); print(f'  {nm:28s} {k:5s} simulated {np.median(v):6.2f} (80% {np.percentile(v,10):.2f}–{np.percentile(v,90):.2f})  actual {act[y1][k]:.2f}  percentile {np.mean(v<act[y1][k])*100:.0f}')
    for y0,y1 in [(2011,2016),(2016,2020),(2011,2020),(2020,2026),(2020,2023),(2023,2026)]:
        print(f'{y0}->{y1}: start t10 {act[y0]["t10"]:.1f}, a100 {act[y0]["a100"]:.2f}')
        rep('equal odds',M.run(y0,years=range(y0,y1),nsim=600),y1); rep('by rank group',M.run(y0,years=range(y0,y1),by_rank=True,nsim=600),y1)
