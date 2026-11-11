import numpy as np, json, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from bridge import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR='#1f5fa8','#d9622b','#2a9d6f','#888888'
out='/home/claude/blog/complexity/millennium-bridge/'
rec=np.load('crowd.npy'); t=rec[:,0]/60
fig,ax=plt.subplots(3,1,figsize=(7.5,6.2),sharex=True,gridspec_kw={'height_ratios':[1,1.3,1.1]})
ax[0].plot(t,rec[:,1],color='k',lw=1.2); ax[0].axhline(Nc,color=C2,ls='--',lw=1); ax[0].text(0.3,Nc+8,f'$N_c$ = {Nc:.0f}',color=C2)
ax[0].set_ylabel('walkers N')
ax[1].plot(t,rec[:,2]*1000,color=C1,lw=1); ax[1].set_ylabel('sway amplitude (mm)')
ax[2].plot(t,rec[:,3],color=C3,lw=.8); ax[2].set_ylabel('walker\norder parameter R'); ax[2].set_xlabel('time (minutes)')
tc=t[np.argmax(rec[:,1]>Nc)]
for a in ax: a.axvline(tc,color=C2,ls=':',lw=1)
fig.tight_layout(); fig.savefig(out+'fig_crowd_test.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
# amplitude vs N
O=json.load(open('oa.json')); S=sorted(json.loads(open(f).read().splitlines()[-1]) for f in glob.glob('st_*.json'))
S=np.array(S); Ns=np.array(O['Ns'])
NN=np.arange(0,401,2); OA=[oa_steady(N) for N in NN]; OA=np.array(OA)
fig,ax=plt.subplots(1,2,figsize=(10,3.7))
ax[0].plot(NN,OA[:,0]*1000,color='k',lw=1.4,label='Ott–Antonsen reduction'); ax[0].plot(S[:,0],S[:,1]*1000,'o',color=C1,label='simulation, N walkers')
ax[0].set_xlabel('walkers N'); ax[0].set_ylabel('sway amplitude (mm)')
ax[1].plot(NN,OA[:,1],color='k',lw=1.4); ax[1].plot(S[:,0],S[:,2],'o',color=C3)
ax[1].set_xlabel('walkers N'); ax[1].set_ylabel('walker order parameter R')
for a in ax: a.axvline(Nc,color=C2,ls='--',lw=1)
ax[0].legend(loc='upper left',fontsize=8); ax[1].text(Nc+5,0.05,'$N_c$',color=C2)
ax[1].annotate('finite-N\nfluctuations ~ $1/\\sqrt{N}$',xy=(120,0.08),xytext=(15,0.35),fontsize=8,color=GR,arrowprops=dict(arrowstyle='->',color=GR))
fig.tight_layout(); fig.savefig(out+'fig_amplitude.png'); plt.close(fig)
# force vs velocity at N=240, steady state
N=240; rng=np.random.default_rng(1); i=np.arange(1,N+1); Om=W0+D*np.tan(np.pi*(i/(N+1)-0.5))
th=rng.uniform(0,2*np.pi,N); X=1e-3; V=0.0; dt=0.004; ts=[];Fs=[];Vs=[];Xs=[]
for k in range(int(1200/dt)):
    A=np.hypot(X,V/W0); Psi=np.arctan2(X,V/W0); F=G*np.sin(th).sum()
    acc=(F-2*M*zeta*W0*V-K*X)/M; V+=dt*acc; X+=dt*V; th+=dt*(Om+C*A*np.sin(Psi-th+alpha))
    if k*dt>1190: ts.append(k*dt-1190); Fs.append(F); Vs.append(V); Xs.append(X)
ts,Fs,Vs,Xs=map(np.array,(ts,Fs,Vs,Xs))
fig,a=plt.subplots(figsize=(7.5,3.2)); b=a.twinx()
a.plot(ts,Xs*1000,color=GR,lw=1.2,label='deck displacement (mm)'); a.plot(ts,Vs*1000/W0,color=C1,lw=1.5,label='deck velocity ÷ Ω₀ (mm)')
b.plot(ts,Fs,color=C2,lw=1.2,label='crowd sideways force (N)'); b.spines['right'].set_visible(True); b.set_ylabel('crowd force (N)',color=C2); b.grid(False)
a.set_xlabel('time (s)'); a.set_ylabel('mm'); a.set_xlim(0,6)
h1,l1=a.get_legend_handles_labels(); h2,l2=b.get_legend_handles_labels(); a.legend(h1+h2,l1+l2,fontsize=8,loc='lower left',ncol=3,bbox_to_anchor=(0,1.0))
fig.tight_layout(); fig.savefig(out+'fig_force_velocity.png'); plt.close(fig)
c=np.corrcoef(Fs,Vs)[0,1]; cx=np.corrcoef(Fs,Xs)[0,1]; print('corr F-V',c,'F-X',cx,'F amp',np.std(Fs)*np.sqrt(2),'V amp',np.std(Vs)*np.sqrt(2),'k per person',np.std(Fs)/np.std(Vs)/N)
