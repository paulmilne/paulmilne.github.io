import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from search import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/probability/endurance-search/'
rng=np.random.default_rng(4)
# F1 drift paths
fig,ax=plt.subplots(1,2,figsize=(10,4.4),sharex=True,sharey=True)
t=np.linspace(0,4,161); dt=t[1]-t[0]
drift=np.array([-3.0,-14.0])   # mean drift per day (km), assumed for illustration: north-westward
for a,two in zip(ax,(False,True)):
    for k in range(60):
        W=np.cumsum(rng.normal(0,np.sqrt(Q*dt),(len(t),2)),axis=0); W[0]=0
        if two: W=W-np.outer(t/4,W[-1])          # Brownian bridge: pinned at the 22 Nov fix
        path=np.outer(t,drift)+W
        a.plot(path[:,0],path[:,1],color=GR,lw=.5,alpha=.5)
        i=np.searchsorted(t,3.0); a.plot(*path[i],'o',color=C2,ms=3,alpha=.8)
    a.plot(0,0,'s',color='k',ms=7); a.text(1.5,1,'fix, 18 Nov',fontsize=8.5)
    if two: a.plot(*(4*drift),'s',color='k',ms=7); a.text(4*drift[0]+1.5,4*drift[1]-3,'fix, 22 Nov',fontsize=8.5)
    a.set_title('only the 18 November fix' if not two else 'fixes on 18 and 22 November',fontsize=9.5); a.set_aspect('equal'); a.set_xlabel('km east')
ax[0].plot([],[],'o',color=C2,label='where the ship is on 21 November'); ax[0].legend(fontsize=8,loc='lower left'); ax[0].set_ylabel('km north')
fig.tight_layout(); fig.savefig(out+'fig_drift.png'); plt.close(fig)
# F2 priors with wreck
X,Y=grid(); fig,ax=plt.subplots(1,2,figsize=(10,4.6),sharey=True)
for a,two in zip(ax,(False,True)):
    s=prior_sigma(two); P=prior(s,X,Y)
    im=a.pcolormesh(X,Y,P,cmap='Blues',shading='nearest'); a.set_aspect('equal'); a.grid(False)
    for k in (1,2): a.add_patch(plt.Circle((0,0),k*s,fill=False,color='k',ls=':',lw=.8))
    a.plot(0,0,'+',color='k',ms=10,mew=2); a.plot(*WRECK,'x',color=C2,ms=10,mew=2.5)
    a.set_title(f'{"one fix" if not two else "two fixes"}: σ = {s:.1f} km',fontsize=9.5); a.set_xlabel('km east of Worsley\'s position')
ax[0].set_ylabel('km north'); ax[1].annotate('wreck, found 2022\n9.4 km south-east',WRECK,xytext=(5,-17),fontsize=8.5,color=C2,arrowprops=dict(arrowstyle='->',color=C2))
fig.tight_layout(); fig.savefig(out+'fig_prior.png'); plt.close(fig)
# F3 posterior evolution (greedy) for the actual wreck, deterministic misses until found
s=prior_sigma(True); P=prior(s,X,Y); B=3; n=P.shape[0]//B; pd=0.9
ij=(int((WRECK[1]+20)//1),int((WRECK[0]+20)//1))
snaps={}; p=P.copy(); found=None
for k in range(1,200):
    M=p[:n*B,:n*B].reshape(n,B,n,B).sum(axis=(1,3)); bi,bj=np.unravel_index(np.argmax(M),M.shape)
    sl=(slice(bi*B,(bi+1)*B),slice(bj*B,(bj+1)*B))
    if sl[0].start<=ij[0]<sl[0].stop and sl[1].start<=ij[1]<sl[1].stop: found=k; break
    mass=p[sl].sum(); p[sl]*=(1-pd); p/=(1-pd*mass)
    if k in (8,20): snaps[k]=p.copy()
print('greedy finds actual wreck (no unlucky misses) at dive',found)
fig,ax=plt.subplots(1,3,figsize=(11,3.9),sharey=True)
for a,(k,pp) in zip(ax,[(0,P)]+sorted(snaps.items())):
    a.pcolormesh(X,Y,pp,cmap='Blues',shading='nearest'); a.set_aspect('equal'); a.grid(False)
    a.plot(*WRECK,'x',color=C2,ms=9,mew=2.5); a.set_title(f'after {k} unsuccessful dives' if k else 'prior',fontsize=9.5); a.set_xlabel('km east')
ax[0].set_ylabel('km north')
fig.tight_layout(); fig.savefig(out+'fig_posterior.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
# F4 dives needed
res={}
for two in (True,False):
    s=prior_sigma(two); P0=prior(s,X,Y); Pf=P0.ravel()
    for strat in ('greedy','spiral'):
        d=[]
        for _ in range(500):
            k=rng.choice(Pf.size,p=Pf); d.append(search(P0,np.unravel_index(k,X.shape),strategy=strat,rng=rng))
        res[(two,strat)]=np.array(d)
fig,a=plt.subplots(figsize=(7,3.8))
for (two,strat),c,ls in (((True,'greedy'),C1,'-'),((True,'spiral'),C1,'--'),((False,'greedy'),C2,'-'),((False,'spiral'),C2,'--')):
    d=np.sort(res[(two,strat)]); a.plot(d,np.arange(1,len(d)+1)/len(d),color=c,ls=ls,lw=1.8,label=f'{"two fixes" if two else "one fix"}, {"most probable block first" if strat=="greedy" else "spiral out from the centre"}')
a.set_xlabel('dives'); a.set_ylabel('probability the wreck is found'); a.set_xlim(0,250); a.legend(fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(out+'fig_dives.png'); plt.close(fig)
for kk,v in res.items(): print(kk,'median',np.median(v),'mean',v.mean().round(1),'p90',np.percentile(v,90))
