"""Prior for the Endurance wreck from two navigational fixes and ice drift, and Bayesian search."""
import numpy as np
FIX=2.0          # km, 1-sigma error of each of Worsley's fixes (assumed)
Q=33.0           # km^2/day, growth of drift uncertainty between fixes (assumed)
T1,T2,TS=0.0,4.0,3.0   # days: fix 18 Nov, fix 22 Nov, sinking 21 Nov
WRECK=np.array([6.65,-6.65])   # km east, north of Worsley's estimated sinking position (9.4 km SE)
def prior_sigma(two_fixes=True):
    if two_fixes:
        bridge=Q*(TS-T1)*(T2-TS)/(T2-T1)
        w1=(T2-TS)/(T2-T1); w2=(TS-T1)/(T2-T1)
        return np.sqrt(bridge+(w1**2+w2**2)*FIX**2)
    return np.sqrt(Q*(TS-T1)+FIX**2)
def grid(L=40,dx=1.0):
    x=np.arange(-L/2+dx/2,L/2,dx); X,Y=np.meshgrid(x,x); return X,Y
def prior(sig,X,Y):
    p=np.exp(-(X**2+Y**2)/(2*sig*sig)); return p/p.sum()
def blocks(X,B=3):
    n=X.shape[0]//B; return n
def search(p,true_ij,pd=0.9,B=3,strategy='greedy',rng=None,max_dives=400):
    """Each dive searches a BxB block of cells. Returns number of dives to detect the wreck."""
    rng=rng or np.random.default_rng()
    p=p.copy(); n=p.shape[0]//B
    order=None
    if strategy=='spiral':
        c=(n-1)/2; idx=[(i,j) for i in range(n) for j in range(n)]
        idx.sort(key=lambda ij: (max(abs(ij[0]-c),abs(ij[1]-c)), np.arctan2(ij[0]-c,ij[1]-c)))
        order=idx*4
    for k in range(max_dives):
        if strategy=='greedy':
            M=p[:n*B,:n*B].reshape(n,B,n,B).sum(axis=(1,3)); bi,bj=np.unravel_index(np.argmax(M),M.shape)
        else: bi,bj=order[k]
        sl=(slice(bi*B,(bi+1)*B),slice(bj*B,(bj+1)*B))
        inside=(sl[0].start<=true_ij[0]<sl[0].stop) and (sl[1].start<=true_ij[1]<sl[1].stop)
        if inside and rng.random()<pd: return k+1
        mass=p[sl].sum(); p[sl]*=(1-pd); p/= (1-pd*mass)
    return max_dives
if __name__=='__main__':
    print('sigma two fixes %.1f km, one fix %.1f km'%(prior_sigma(True),prior_sigma(False)))
    X,Y=grid(); s=prior_sigma(True)
    ij=(int((WRECK[1]+20)//1),int((WRECK[0]+20)//1))
    rng=np.random.default_rng(0)
    for strat in ('greedy','spiral'):
        d=[search(prior(s,X,Y),ij,strategy=strat,rng=rng) for _ in range(300)]
        print(strat,'dives to find actual wreck: median',np.median(d),'mean',np.mean(d))
    # expected over prior draws
    for strat in ('greedy','spiral'):
        P=prior(s,X,Y).ravel(); d=[]
        for _ in range(400):
            k=rng.choice(P.size,p=P); d.append(search(prior(s,X,Y),np.unravel_index(k,X.shape),strategy=strat,rng=rng))
        print(strat,'over the prior: median',np.median(d),'mean',np.mean(d),'90%',np.percentile(d,90))
