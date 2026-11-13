import numpy as np, time
from scipy.integrate import solve_ivp
from kerr import camera_rays, deriv, trace
a=0.6; r0=25.0; th0=np.radians(85); rh=1+np.sqrt(1-a*a)
pr,pth,L,E=camera_rays(40,20,60,r0,th0,a)
def ref(j,i,r_far=60):
    def f(lam,y): d=deriv(y[0],y[1],y[3],y[4],L[j,i],a,lambda x:x); return [d[0],d[1],d[2],d[3],d[4]]
    hor=lambda lam,y: y[0]-rh*1.01; hor.terminal=True
    far=lambda lam,y: y[0]-r_far; far.terminal=True
    s=solve_ivp(f,[0,1e4],[r0,th0,0,pr[j,i],pth[j,i]],method='DOP853',rtol=1e-12,atol=1e-12,events=[hor,far])
    return (0 if s.t_events[1].size else 1), s.y[1,-1], s.y[2,-1]
def angerr(t1,p1,t2,p2):
    v=lambda t,p: np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    return np.arccos(np.clip(v(t1,p1)@v(t2,p2),-1,1))
for h0 in (0.05,0.02,0.01):
    out=trace(pr,pth,L,r0,th0,a,np.float64,h0=h0)
    rng=np.random.default_rng(0); sample=[(rng.integers(20),rng.integers(40)) for _ in range(40)]
    errs=[];agree=0
    for j,i in sample:
        fr,t,p=ref(j,i); agree+=fr==out['fate'][j,i]
        if fr==0 and out['fate'][j,i]==0: errs.append(angerr(t,p,out['th'][j,i],out['ph'][j,i]))
    errs=np.array(errs); print(f'h0={h0}: fate agree {agree}/40, escaped-ray angular error median {np.median(errs):.2e} max {errs.max():.2e} rad; mean steps {out["steps"].mean():.0f}')
