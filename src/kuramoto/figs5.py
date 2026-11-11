import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ff import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/complexity/esp32-fireflies/'
# F1 model
fig,ax=plt.subplots(1,2,figsize=(9.5,3.6))
p=np.linspace(0,1,300); a=ax[0]; a.plot(p,f(p),color=C1,lw=2); a.set_xlabel('phase φ'); a.set_ylabel('state x = f(φ)')
eps=0.15
for p0,c in ((0.25,C3),(0.75,C2)):
    x0=f(p0); a.annotate('',xy=(p0,x0+eps),xytext=(p0,x0),arrowprops=dict(arrowstyle='->',color=c,lw=1.6))
    p1=finv(x0+eps); a.plot([p0,p1],[x0+eps,x0+eps],':',color=c); a.plot([p1],[x0+eps],'o',color=c,ms=4)
    a.text(p0-0.02,x0+eps/2,'+ε',ha='right',color=c,fontsize=9)
    a.annotate('',xy=(p1,-0.06),xytext=(p0,-0.06),arrowprops=dict(arrowstyle='->',color=c,lw=1.2),annotation_clip=False)
a.text(0.5,0.08,'same kick, bigger\nphase jump late in the cycle',fontsize=8,color=GR,ha='center')
a.axhline(1,color='k',lw=.8,ls='--'); a.text(0.02,1.02,'threshold: flash',fontsize=8); a.set_ylim(-0.08,1.12); a.set_title('state against phase (ε = 0.15 shown)',fontsize=9)
a=ax[1]
for e,c in ((0.02,C1),(0.05,C3),(0.15,C2)):
    x=f(p)+e; dp=np.where(x>=1,1-p,finv(np.minimum(x,1))-p); a.plot(p,dp,color=c,lw=1.6,label=f'ε = {e}')
a.set_xlabel('phase when the flash arrives'); a.set_ylabel('phase advance Δφ'); a.set_title('phase response curve',fontsize=9); a.legend(fontsize=8,loc='upper left')
a.text(0.8,0.3,'late flashes push\nit straight to\nthreshold: it fires\nat once (absorbed)',fontsize=8,color=GR)
fig.tight_layout(); fig.savefig(out+'fig_model.png'); plt.close(fig)
# F2 raster
fl,rs=sim(eps=0.008,spread=0.0,Tend=12,seed=3)
fig,ax=plt.subplots(2,1,figsize=(8,4.6),sharex=True,gridspec_kw={'height_ratios':[2.4,1]})
ax[0].plot(fl[:,0],fl[:,1],'|',color=BR,ms=7,mew=1.6); ax[0].set_ylabel('node'); ax[0].set_ylim(-1,20); ax[0].grid(False)
ax[1].plot(rs[:,0],rs[:,1],color=C1); ax[1].set_ylabel('order parameter r'); ax[1].set_xlabel('time (s)'); ax[1].set_ylim(0,1.05)
fig.tight_layout(); fig.savefig(out+'fig_raster.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
# F3 delays
cases=[('instant pulses',dict(eps=0.05)),('3 ms delay',dict(eps=0.05,delay=0.003)),('3 ms delay + 50 ms refractory',dict(eps=0.05,delay=0.003,refr=0.05)),
       ('radio-like: 3–5 ms delay, 10% loss,\n2% clock spread, 50 ms refractory',dict(eps=0.05,delay=0.003,jitter=0.002,loss=0.1,spread=0.02,refr=0.05))]
fig,ax=plt.subplots(2,4,figsize=(12,4.6),sharex=True)
for j,(name,kw) in enumerate(cases):
    fl,rs=sim(Tend=30,seed=2,**kw)
    ax[0,j].plot(rs[:,0],rs[:,1],color=C1,lw=.8); ax[0,j].set_ylim(0,1.05); ax[0,j].set_title(name,fontsize=8.5)
    h,e=np.histogram(fl[:,0],bins=np.arange(0,30.01,3)); ax[1,j].step(e[:-1],h/3,where='post',color=C2); ax[1,j].set_xlabel('time (s)')
    ax[1,j].set_yscale('log'); ax[1,j].set_ylim(5,1000)
ax[0,0].set_ylabel('order parameter r'); ax[1,0].set_ylabel('flashes per second\n(all 20 nodes)')
fig.tight_layout(); fig.savefig(out+'fig_delay.png'); plt.close(fig)
# F4 topology
T=json.load(open('topo.json'))
xy=np.array([(i%5,i//5) for i in range(20)],float)
fig,ax=plt.subplots(1,2,figsize=(9.5,3.6),gridspec_kw={'width_ratios':[1,1.4]})
a=ax[0]; a.set_aspect('equal'); a.axis('off')
for (x,y) in xy: a.plot(x,y,'o',color=BR,ms=9)
for rad,c,ls in ((1.01,C1,'-'),(1.5,C3,'--'),(2.3,C2,':')):
    a.add_patch(plt.Circle((2,1),rad,fill=False,color=c,ls=ls,lw=1.3))
a.plot(2,1,'o',color='k',ms=9); a.set_xlim(-0.6,4.6); a.set_ylim(-1.5,3.6); a.set_title('20 nodes, 5 × 4, and three sensing ranges',fontsize=9)
a=ax[1]; labs=['neighbours\n(range 1)','+ diagonals\n(range 1.5)','range 2.3','everyone']
for k,rad in enumerate((1.01,1.5,2.3,0)):
    v=[x[2] for x in T if x[0]==rad]; ok=[x for x in v if x is not None]; nf=len(v)-len(ok)
    a.plot(np.full(len(ok),k)+np.linspace(-.15,.15,len(ok)),ok,'o',color=[C1,C3,C2,'k'][k],ms=4)
    if nf: a.text(k,185,f'{nf} of 12 not\nsynced in 3 min',ha='center',fontsize=7.5,color=GR)
    a.plot([k-.25,k+.25],[np.median([x if x is not None else 999 for x in v])]*2,color='k',lw=1.5)
a.set_yscale('log'); a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8); a.set_ylabel('time to synchronise (s)'); a.set_ylim(0.5,300)
fig.tight_layout(); fig.savefig(out+'fig_topology.png'); plt.close(fig)
# F5 map
G=json.load(open('grid.json')); E=sorted(set(g[0] for g in G)); S=sorted(set(g[1] for g in G))
Mx=np.zeros((len(E),len(S)))
for e,sp,s,rm,rmin in G: Mx[E.index(e),S.index(sp)]+=(rmin>0.9)/3
fig,a=plt.subplots(figsize=(6.4,4))
im=a.pcolormesh(np.array(S)*100,E,Mx,cmap='Blues',shading='nearest',vmin=0,vmax=1); a.set_yscale('log')
a.set_xlabel('spread of natural periods (%)'); a.set_ylabel('kick per received flash ε'); a.grid(False)
cb=fig.colorbar(im,ax=a); cb.set_label('fraction of runs synchronised')
fig.tight_layout(); fig.savefig(out+'fig_map.png'); plt.close(fig)
