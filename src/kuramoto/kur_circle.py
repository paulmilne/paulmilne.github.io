import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'figure.dpi':150})
g=0.5; N=400; i=np.arange(1,N+1); w=g*np.tan(np.pi*(i/(N+1)-0.5))
fig,ax=plt.subplots(1,3,figsize=(9,3.3))
for a,K in zip(ax,(0.6,1.3,2.5)):
    rng=np.random.default_rng(2); th=rng.uniform(0,2*np.pi,N); dt=0.02
    for k in range(int(200/dt)):
        z=np.exp(1j*th).mean(); th=th+dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
    z=np.exp(1j*th).mean(); th=th-np.angle(z)
    lock=abs(w)<K*abs(z)
    a.add_patch(plt.Circle((0,0),1,fill=False,color='#bbbbbb'))
    a.scatter(np.cos(th[~lock]),np.sin(th[~lock]),s=6,color='#888888',alpha=.6,label='drifting')
    a.scatter(np.cos(th[lock]),np.sin(th[lock]),s=6,color='#1f5fa8',label='locked')
    a.annotate('',xy=(abs(z),0),xytext=(0,0),arrowprops=dict(arrowstyle='->',color='#d9622b',lw=2))
    a.set_title(f'K = {K}:  r = {abs(z):.2f}, {lock.mean()*100:.0f}% locked',fontsize=9)
    a.set_xlim(-1.2,1.2); a.set_ylim(-1.2,1.2); a.set_aspect('equal'); a.axis('off')
ax[2].legend(loc='lower right',fontsize=8,frameon=False,markerscale=2)
fig.tight_layout(); fig.savefig('fig_circle.png'); fig.savefig('thumbnail.png')
