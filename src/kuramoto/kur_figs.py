import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR='#1f5fa8','#d9622b','#2a9d6f','#888888'
out='/home/claude/blog/complexity/kuramoto-synchronisation/'
def lorentz_q(N,g): i=np.arange(1,N+1); return g*np.tan(np.pi*(i/(N+1)-0.5))
def gauss_q(N,s):
    from scipy.stats import norm; i=np.arange(1,N+1); return s*norm.ppf(i/(N+1))
def ramp(w,Kmax=4,T=800,dt=0.02,seed=0):
    rng=np.random.default_rng(seed); th=rng.uniform(0,2*np.pi,len(w)); n=int(T/dt); Ks=[];rs=[]
    for k in range(n):
        K=Kmax*k/n; z=np.exp(1j*th).mean(); r=abs(z); psi=np.angle(z)
        th=th+dt*(w+K*r*np.sin(psi-th))
        if k%25==0: Ks.append(K); rs.append(r)
    return np.array(Ks),np.array(rs)
g=0.5; s=0.5
KK=np.linspace(0,4,400)
fig,ax=plt.subplots(1,2,figsize=(9,3.6),sharey=True)
for a,name,wfun,par in ((ax[0],'Lorentzian, γ = 0.5',lorentz_q,g),(ax[1],'Gaussian, σ = 0.5',gauss_q,s)):
    for N,c,al in ((100,GR,.7),(2000,C1,.9)):
        K,r=ramp(wfun(N,par)); a.plot(K,r,color=c,lw=.8,alpha=al,label=f'simulation, N = {N}')
    if wfun is lorentz_q:
        Kc=2*g; th=np.where(KK>Kc,np.sqrt(np.clip(1-Kc/KK,0,1)),0)
    else:
        gw=lambda w: np.exp(-w*w/(2*s*s))/(s*np.sqrt(2*np.pi)); Kc=2/(np.pi*gw(0))
        def rsol(K):
            if K<=Kc: return 0.0
            f=lambda r: K/2*quad(lambda u: gw(K*r*np.sin(u))*np.cos(u)**2,-np.pi,np.pi)[0]-1
            return brentq(f,1e-6,1)
        th=np.array([rsol(K) for K in KK])
    a.plot(KK,th,color=C2,lw=2,label='theory (self-consistency)')
    a.axvline(Kc,color=GR,ls='--',lw=1); a.text(Kc+0.08,0.08,f'$K_c$ = {Kc:.2f}',color=GR)
    a.set_title(name,fontsize=10); a.set_xlabel('coupling K')
ax[0].set_ylabel('order parameter r'); ax[0].legend(loc='lower right',fontsize=8)
fig.tight_layout(); fig.savefig(out+'fig_r_vs_K.png'); plt.close(fig)
# effective frequency plateau
N=2000; w=lorentz_q(N,g); K=1.6; dt=0.02; rng=np.random.default_rng(1); th=rng.uniform(0,2*np.pi,N)
for k in range(int(300/dt)):
    z=np.exp(1j*th).mean(); th=th+dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
th0=th.copy(); T=200; rr=[]
for k in range(int(T/dt)):
    z=np.exp(1j*th).mean(); rr.append(abs(z)); th=th+dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
weff=(th-th0)/T; r=np.mean(rr)
ww=np.linspace(-3,3,600); theo=np.where(abs(ww)<K*r,0,np.sign(ww)*np.sqrt(np.clip(ww**2-(K*r)**2,0,None)))
fig,a=plt.subplots(figsize=(6.4,3.8))
m=abs(w)<3
a.plot(w[m],weff[m],'.',ms=2.5,color=C1,label='simulation, each oscillator')
a.plot(ww,theo,color=C2,lw=1.5,label=r'theory: 0 if $|\omega|<Kr$, else $\pm\sqrt{\omega^2-(Kr)^2}$')
a.plot(ww,ww,color=GR,ls=':',lw=1,label='uncoupled: own frequency')
a.axvspan(-K*r,K*r,color=C3,alpha=.12); a.text(0,1.6,'locked',ha='center',color=C3)
a.set_xlabel('natural frequency ω'); a.set_ylabel('measured frequency'); a.set_xlim(-3,3); a.set_ylim(-3,3); a.legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_plateau.png'); plt.close(fig)
print('r at K=1.6:',r,'theory',np.sqrt(1-2*g/K),'Kc gauss',2/(np.pi/(s*np.sqrt(2*np.pi))))
