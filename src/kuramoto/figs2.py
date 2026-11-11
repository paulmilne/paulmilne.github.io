import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR='#1f5fa8','#d9622b','#2a9d6f','#888888'
out='/home/claude/blog/complexity/ott-antonsen-bimodal/'
g=0.5
# --- 1. OA check, unimodal
N=4000; i=np.arange(1,N+1); w=g*np.tan(np.pi*(i/(N+1)-0.5)); dt=0.01; T=40
fig,a=plt.subplots(figsize=(6.6,3.6))
for K,c in ((1.3,C3),(2.0,C1),(3.0,C2)):
    rng=np.random.default_rng(3); rho=0.03; u=rng.permutation((np.arange(N)+0.5)/N)
    th=2*np.arctan((1-rho)/(1+rho)*np.tan(np.pi*(u-0.5))); ts=[];rs=[]
    for k in range(int(T/dt)):
        z=np.exp(1j*th).mean(); ts.append(k*dt); rs.append(abs(z)); th+=dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
    a.plot(ts,rs,color=c,lw=.8,alpha=.7)
    r=rs[0]; ro=[]
    for t in ts: ro.append(r); r+=dt*(-g*r+K/2*r*(1-r*r))
    a.plot(ts,ro,'--',color='k',lw=1.1)
    a.text(T+0.5,np.sqrt(1-2*g/K),f'K = {K}',color=c,va='center',fontsize=9)
a.plot([],[],color=GR,label='simulation, N = 4000'); a.plot([],[],'--',color='k',label='Ott–Antonsen: one equation for r')
a.set_xlabel('time'); a.set_ylabel('r'); a.set_xlim(0,T+5); a.legend(fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(out+'fig_oa_check.png'); plt.close(fig)
# --- 2. bimodal: g and bifurcation diagrams
br=json.load(open('branches.json'))
fig,ax=plt.subplots(2,3,figsize=(11,5.6),gridspec_kw={'height_ratios':[1,2.3]})
ww=np.linspace(-3,3,600)
L=lambda x: g/np.pi/(x*x+g*g)
for j,(w0,Kmax) in enumerate(((0.2,4),(0.4,4),(1.0,8))):
    a=ax[0,j]; a.fill_between(ww,0.5*(L(ww-w0)+L(ww+w0)),color=C1,alpha=.25); a.plot(ww,0.5*(L(ww-w0)+L(ww+w0)),color=C1)
    a.set_title(f'({"abc"[j]})  ω₀ = {w0},  ω₀/γ = {w0/g:.1f}',fontsize=10); a.set_yticks([]); a.set_xlabel('ω',labelpad=0); a.set_ylim(0,0.7)
    a=ax[1,j]; d=json.load(open(f'sweep_{w0}.json'))
    B=np.array(br[str(w0)]); st=B[:,2]>0.5
    Kp=2*(g*g+w0*w0)/g; Kinc=Kp if w0<g else 4*g
    a.plot([0,Kinc],[0,0],color='k',lw=2); a.plot([Kinc,Kmax],[0,0],'--',color='k',lw=1.2)
    # split stable/unstable branch
    for s,ls,lw in ((True,'-',2),(False,'--',1.2)):
        Bs=np.where((B[:,2]>0.5)==s,B[:,1],np.nan); a.plot(Bs,B[:,0],ls,color='k',lw=lw)
    for key,c,m,lab in (('full_up',C1,'^','simulation, K rising'),('full_dn',C2,'v','simulation, K falling')):
        A=np.array(d[key]); a.plot(A[:,0],A[:,1],m,ms=3.5,color=c,label=lab,alpha=.85)
        osc=A[:,3]-A[:,2]>0.08; a.vlines(A[osc,0],A[osc,2],A[osc,3],color=c,alpha=.35,lw=1.2)
    a.set_xlim(0,Kmax); a.set_ylim(-0.03,1); a.set_xlabel('K')
    if w0==1.0:
        a.annotate('standing waves:\nr oscillates\n(bars: min–max)',xy=(2.4,0.42),xytext=(0.3,0.55),fontsize=8,color=GR,arrowprops=dict(arrowstyle='->',color=GR))
        a.annotate('unstable branch\nmeets r = 0 at K = 5',xy=(4.9,0.03),xytext=(5.0,0.3),fontsize=8,color=GR,arrowprops=dict(arrowstyle='->',color=GR))
    if w0==0.4:
        a.annotate('bistable',xy=(1.61,0.15),xytext=(2.2,0.15),fontsize=8,color=GR,arrowprops=dict(arrowstyle='->',color=GR))
ax[1,0].plot([],[],'-',color='k',lw=2,label='Ott–Antonsen, stable'); ax[1,0].plot([],[],'--',color='k',lw=1.2,label='Ott–Antonsen, unstable')
ax[1,0].set_ylabel('r'); ax[1,0].legend(fontsize=7.5,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_bimodal.png'); plt.close(fig)
# --- 3. standing waves
w0=1.0; K=2.5; N=2000; h=N//2; uu=np.arange(1,h+1)/(h+1); Lq=g*np.tan(np.pi*(uu-0.5)); w=np.concatenate([Lq+w0,Lq-w0])
rng=np.random.default_rng(4); th=rng.uniform(0,2*np.pi,N); dt=0.01; ts=[];rs=[]; snaps={}
for k in range(int(160/dt)):
    t=k*dt; z=np.exp(1j*th).mean(); ts.append(t); rs.append(abs(z)); th+=dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
rs=np.array(rs); ts=np.array(ts); m=ts>120
pk=ts[m][np.argmax(rs[m])]; tr=ts[m][np.argmin(rs[m])]
# rerun to capture snapshots at those times
rng=np.random.default_rng(4); th=rng.uniform(0,2*np.pi,N)
for k in range(int(160/dt)):
    t=k*dt
    for name,tt in (('max',pk),('min',tr)):
        if abs(t-tt)<dt/2: snaps[name]=th.copy()
    z=np.exp(1j*th).mean(); th+=dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
fig=plt.figure(figsize=(10,3.4)); gs=fig.add_gridspec(1,4)
a=fig.add_subplot(gs[0,:2]); a.plot(ts,rs,color=C1,lw=.8); a.set_xlim(100,160); a.set_xlabel('time'); a.set_ylabel('r')
a.plot([pk],[rs[ts==pk][0]] if (ts==pk).any() else [rs.max()],'o',color=C2); a.plot([tr],[rs[m].min()],'o',color=C3)
for j,(name,c) in enumerate((('max',C2),('min',C3))):
    a=fig.add_subplot(gs[0,2+j]); th=snaps[name]
    a.add_patch(plt.Circle((0,0),1,fill=False,color='#bbbbbb'))
    a.add_patch(plt.Circle((0,0),0.8,fill=False,color='#dddddd'))
    a.scatter(np.cos(th[:h]),np.sin(th[:h]),s=3,color=C1,alpha=.35,label='ω near +ω₀ (outer)')
    a.scatter(0.8*np.cos(th[h:]),0.8*np.sin(th[h:]),s=3,color=C2,alpha=.35,label='ω near −ω₀ (inner)')
    for grp,c2 in ((th[:h],C1),(th[h:],C2)):
        zz=np.exp(1j*grp).mean(); a.annotate('',xy=(zz.real,zz.imag),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=c2,lw=1.8,shrinkA=0,shrinkB=0))
    z=np.exp(1j*th).mean(); a.annotate('',xy=(z.real,z.imag),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='k',lw=2.2,shrinkA=0,shrinkB=0))
    a.set_title(f'r = {abs(z):.2f}',fontsize=9,color=c); a.set_xlim(-1.2,1.2); a.set_ylim(-1.2,1.2); a.set_aspect('equal'); a.axis('off')
    if j==1: a.legend(fontsize=7,loc='lower right',markerscale=3,bbox_to_anchor=(1.15,-0.12))
fig.tight_layout(); fig.savefig(out+'fig_standing.png'); plt.close(fig)
# --- 4. phase diagram
P=np.load('phase.npz'); lo,hi=P['lo'],P['hi']
reg=np.zeros(lo.shape,int)
reg[(lo==1)&(hi==1)]=1; reg[(lo==2)&(hi==2)]=2; reg[(lo==0)&(hi==1)]=3; reg[(lo==2)&(hi==1)]=4
reg[(lo==1)&(hi==0)]=1; reg[(lo==2)&(hi==0)]=2
cm=ListedColormap(['#f2f2f2','#cfe0f3','#f7d9c4','#c9e8da','#e8d3ee'])
fig,a=plt.subplots(figsize=(7.2,4.8))
a.pcolormesh(P['W'],P['K'],reg,cmap=cm,vmin=-.5,vmax=4.5,shading='nearest')
x=np.linspace(0,1,100); a.plot(x,2*(1+x*x),color='k',lw=1.5); a.plot([1,2.2],[4,4],color='k',lw=1.5,ls=':')
F=json.load(open('fold.json')); xf=np.array(F['x']); kf=np.array(F['Ksn']); sel=xf>1/np.sqrt(3)-0.02
a.plot(xf[sel],kf[sel],color=C2,lw=1.5)
a.axvline(1/np.sqrt(3),color=GR,lw=.8,ls='--'); a.text(1/np.sqrt(3)+0.02,0.75,'g has two peaks\nfor ω₀/γ > 1/√3',fontsize=7.5,color=GR)
for xx,lab_ in ((0.4,'a'),(0.8,'b'),(2.0,'c')):
    a.plot([xx],[0.08],'v',color='k',ms=6,clip_on=False); a.text(xx,0.35,lab_,ha='center',fontsize=9,weight='bold')
L_=dict(fontsize=9,ha='center')
a.text(0.45,1.3,'incoherent',**L_); a.text(0.6,6,'partially synchronised',**L_); a.text(1.75,4.65,'standing waves',**L_)
a.annotate('bistable: incoherent\nor partially synchronised',xy=(0.88,3.4),xytext=(0.05,4.3),fontsize=8,arrowprops=dict(arrowstyle='->',color=GR))
a.annotate('bistable: standing waves\nor partially synchronised',xy=(1.15,4.3),xytext=(1.25,7.0),fontsize=8,arrowprops=dict(arrowstyle='->',color=GR))
a.text(0.03,2.3,'pitchfork  K = 2(γ²+ω₀²)/γ',fontsize=7.5,rotation=7); a.text(1.85,3.7,'Hopf  K = 4γ',fontsize=7.5)
a.text(1.55,5.25,'saddle-node',fontsize=7.5,color=C2,rotation=24)
a.set_xlabel('peak separation ω₀/γ'); a.set_ylabel('coupling K/γ'); a.set_xlim(0,2.2); a.set_ylim(0,8); a.grid(False)
fig.tight_layout(); fig.savefig(out+'fig_phase.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
print('pk',pk,'tr',tr)
