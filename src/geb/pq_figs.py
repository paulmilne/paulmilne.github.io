"""pq-system figures: the theorems as a lattice of derivations, and how rare theorems are."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from math import comb
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='../../geb/pq-system/'
s=lambda x,y,z:'-'*x+'p'+'-'*y+'q'+'-'*z

# F1: every theorem, placed by (x, y); arrows show the rule
N=6
fig,ax=plt.subplots(figsize=(9,6.2))
for x in range(1,N+1):
    for y in range(1,N+1):
        ax.plot(x,y,'o',color=C2 if y==1 else C1,ms=7,zorder=3)
        if y<N: ax.annotate('',(x,y+1-.12),(x,y+.12),arrowprops=dict(arrowstyle='-|>',color=GR,lw=.9))
for (x,y) in [(2,3),(1,1),(4,2),(3,5)]:
    ax.annotate(f'{s(x,y,x+y)}\n{x} + {y} = {x+y}',(x,y),xytext=(x+.14,y+.18),fontsize=7.5,family='monospace',
                bbox=dict(fc='white',ec='#ccc',pad=2))
ax.set_xlim(.5,N+1.4); ax.set_ylim(.4,N+.9)
ax.set_xlabel('hyphens before p   (x)'); ax.set_ylabel('hyphens between p and q   (y)')
ax.plot([],[],'o',color=C2,label='axiom:  x p - q x-'); ax.plot([],[],'o',color=C1,label='theorem from the rule'); ax.plot([],[],color=GR,label='rule: one more hyphen in the middle and last groups')
ax.legend(fontsize=8.5,loc='upper left',ncol=3,bbox_to_anchor=(0,1.08))
fig.tight_layout(); fig.savefig(out+'fig_theorems.png'); plt.close(fig)

# F2: how rare theorems are among well-formed strings
n=np.arange(5,61)
wf=np.array([comb(k-3,2) for k in n])                       # x, y, z each at least one hyphen
th=np.array([((k-2)//2-1) if k%2==0 else 0 for k in n])     # 2(x+y)+2 = n
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].semilogy(n,wf,color=GR,label='well-formed strings')
ax[0].semilogy(n[th>0],th[th>0],'o',color=C1,ms=3,label='theorems')
ax[0].semilogy(n,3.0**n,':',color=BR,label='all strings of p, q and -')
ax[0].set_ylim(1,1e12); ax[0].set_xlabel('length of string'); ax[0].set_ylabel('number'); ax[0].legend(fontsize=8); ax[0].set_title('Counting strings by length',fontsize=9.5)
ev=n%2==0
ax[1].plot(n[ev],th[ev]/wf[ev],'o-',color=C1,ms=3,label='theorems / well-formed')
ax[1].plot(n[ev],1/n[ev],'--',color=C2,label='≈ 1 / length')
ax[1].set_xlabel('length of string (even lengths)'); ax[1].set_ylabel('share that are theorems'); ax[1].legend(fontsize=8); ax[1].set_title('Theorems become rare',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_rare.png'); plt.close(fig)
print([(k,a,b) for k,a,b in zip(n,wf,th)][:8])

# thumbnail
fig,ax=plt.subplots(figsize=(6,4)); ax.axis('off'); fig.patch.set_facecolor('#14243b')
ax.text(.5,.66,'--p---q-----',ha='center',va='center',fontsize=30,family='monospace',color='white')
ax.text(.5,.38,'2  +  3  =  5',ha='center',va='center',fontsize=22,family='monospace',color='#9fc3ee')
ax.text(.5,.12,'GEB · Part 2',ha='center',fontsize=11,color='#c9a85c')
fig.savefig(out+'thumbnail.png',facecolor=fig.get_facecolor()); plt.close(fig)
