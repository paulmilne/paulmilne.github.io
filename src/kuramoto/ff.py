"""Pulse-coupled (Mirollo-Strogatz) oscillators with delay, loss, refractory period and limited range."""
import numpy as np
b=3.0
def f(p): return np.log1p((np.exp(b)-1)*p)/b          # concave state from phase
def finv(x): return np.expm1(b*x)/(np.exp(b)-1)
def sim(N=20,T0=1.0,spread=0.0,eps=0.05,delay=0.0,loss=0.0,refr=0.0,adj=None,Tend=60,dt=5e-4,seed=0,phi0=None,jitter=0.0):
    rng=np.random.default_rng(seed)
    T=T0*(1+spread*rng.standard_normal(N)); w=1/T
    phi=rng.random(N) if phi0 is None else phi0.copy()
    since=np.full(N,1e9)                              # time since own flash
    if adj is None: adj=~np.eye(N,dtype=bool)
    pend=[]                                           # (arrival_time, receiver mask)
    flashes=[]; rs=[]; t=0.0
    n=int(Tend/dt)
    for k in range(n):
        phi+=w*dt; since+=dt
        fired=phi>=1
        # deliver pulses due now
        rec=np.zeros(N)
        keep=[]
        for (ta,mask) in pend:
            if ta<=t: rec+=mask
            else: keep.append((ta,mask))
        pend=keep
        if rec.any():
            ok=(rec>0)&(since>=refr)&(~fired)
            if ok.any():
                x=f(np.clip(phi[ok],0,1))+eps*rec[ok]
                newfire=x>=1
                p=phi[ok]; p=np.where(newfire,1.0,finv(np.minimum(x,1))); phi[ok]=p
                fired=phi>=1
        if fired.any() and delay==0 and jitter==0:
            # instantaneous pulses: cascade within this instant (Mirollo-Strogatz absorption)
            done=np.zeros(N,bool); new=fired.copy()
            while new.any():
                done|=new
                rec=np.zeros(N)
                for i in np.where(new)[0]:
                    flashes.append((t,i)); rec+=adj[i]&(rng.random(N)>=loss)
                ok=(rec>0)&~done&(since>=refr)
                x=f(np.clip(phi[ok],0,1))+eps*rec[ok]; phi[ok]=np.where(x>=1,1.0,finv(np.minimum(x,1)))
                new=(phi>=1)&~done
            phi[done]=0.0; since[done]=0.0; fired[:]=False
        if fired.any():
            idx=np.where(fired)[0]
            for i in idx:
                flashes.append((t,i))
                recv=adj[i]&(rng.random(N)>=loss)
                d=delay+jitter*rng.random()
                if d<=0: pend.append((t,recv.astype(float)))   # delivered next step
                else: pend.append((t+d,recv.astype(float)))
            phi[fired]=0.0; since[fired]=0.0
        if k%20==0: rs.append((t,abs(np.exp(2j*np.pi*phi).mean())))
        t+=dt
    return np.array(flashes),np.array(rs)
