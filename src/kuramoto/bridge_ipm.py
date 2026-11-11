import numpy as np, json, sys
from ipm import g,L,Om,bmin,m
M=1.13e5; W0=2*np.pi*1.03; K=M*W0**2; zeta=0.0075
F=json.load(open('fpdep.json')); fpx=np.array([r[0] for r in F]); dc=np.array([r[1] for r in F])
from scipy.stats import norm
xs=np.linspace(0.6,1.25,400); wgt=norm.pdf(xs,0.935,0.093); dC=np.sum(np.interp(xs,fpx,dc,left=dc[0],right=dc[-1])*wgt)/wgt.sum()
Nc=2*M*zeta*W0/(-dC); print('mean dC',dC,'Nc',Nc)
def run(N,Tend=600,dt=2e-3,seed=0):
    rng=np.random.default_rng(seed); fp=rng.normal(0.935,0.093,N); T=1/(2*fp); w=bmin*(1+np.exp(Om*T))
    s=np.where(rng.random(N)<.5,1.,-1.); tn=rng.random(N)*T            # random gait phases
    # start each walker near its periodic orbit: COM between feet
    urel=s*w/2; y=-s*w/2*0.2; v=np.zeros(N); X=1e-3; V=0.0; out=[]; t=0.0; ph=np.zeros(N)
    last=np.zeros(N)
    for k in range(int(Tend/dt)):
        due=t>=tn
        if due.any():
            urel[due]=(y[due]-X)+v[due]/Om+s[due]*bmin; s[due]=-s[due]; last[due]=tn[due]; tn[due]+=T[due]
        a=Om**2*(y-(X+urel)); Fb=-m*a.sum()
        Ab=(Fb-2*M*zeta*W0*V-K*X)/M
        V+=dt*Ab; X+=dt*V; v+=dt*a; y+=dt*v; t+=dt
        if np.hypot(X,V/W0)>0.3: break
        if k%50==0:
            gait=np.pi*(t-last)/T+np.where(s>0,np.pi,0)            # lateral gait phase
            out.append((t,np.hypot(X,V/W0),abs(np.exp(1j*gait).mean()),Fb))
    return np.array(out),Nc
if __name__=='__main__':
    for N in (int(sys.argv[1]),):
        o,_=run(N); np.save(f'bip_{N}.npy',o); print(N,'A start/end mm',o[100,1]*1e3,o[-1,1]*1e3,'R mean',o[:,2].mean())
