import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from mu import successors, bfs
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
RC={1:C1,2:C2,3:C3,4:BR}
out='../../geb/mu-puzzle/'

# F1: the first three generations of theorems
levels=[["I"]]; seen={"I"}; edges=[]
for g in range(3):
    nxt=[]
    for s in levels[-1]:
        for r,i,t in successors(s):
            edges.append((s,t,r))
            if t not in seen: seen.add(t); nxt.append(t)
    levels.append(nxt)
fig,ax=plt.subplots(figsize=(10,4.6)); ax.axis('off'); ax.grid(False)
pos={}
for g,lv in enumerate(levels):
    for k,s in enumerate(lv):
        pos[s]=((k+.5)/len(lv), 1-g/3)
for s,t,r in edges:
    (x0,y0),(x1,y1)=pos[s],pos[t]
    back=y1>=y0
    if back:
        ax.add_patch(FancyArrowPatch((x0+.02,y0+.045),(x1+.04,y1-.04),arrowstyle='-|>',mutation_scale=9,color=RC[r],lw=1.1,ls='--',
            connectionstyle='arc3,rad=.3',shrinkA=4,shrinkB=4))
        ax.text((x0+x1)/2,y1+.11,'MIIII → MIU: back to step 1',fontsize=7.5,color=RC[r],ha='center')
    else:
        ax.add_patch(FancyArrowPatch((x0,y0-.04),(x1,y1+.045),arrowstyle='-|>',mutation_scale=9,color=RC[r],lw=1.1,shrinkA=6,shrinkB=6,alpha=.9))
for s,(x,y) in pos.items():
    ax.text(x,y,'M'+s,ha='center',va='center',fontsize=9 if len(s)<9 else 7.5,family='monospace',
            bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#444',lw=.8))
for r,lab in [(1,'I:  xI → xIU'),(2,'II:  Mx → Mxx'),(3,'III:  III → U'),(4,'IV:  UU → (deleted)')]:
    ax.plot([],[],color=RC[r],lw=2,label=lab)
ax.legend(loc='upper left',fontsize=8.5,ncol=1,prop={'family':'monospace','size':8.5})
for g in range(4): ax.text(1.0,1-g/3,f'step {g}',fontsize=8,color=GR,ha='right',va='bottom')
ax.set_xlim(-.02,1.02); ax.set_ylim(-.08,1.08)
fig.tight_layout(); fig.savefig(out+'fig_tree.png'); plt.close(fig)

# F2: the invariant as a three-state machine
fig,ax=plt.subplots(figsize=(7.5,3.2)); ax.axis('off'); ax.grid(False); ax.set_aspect('equal')
P={1:(0,0),2:(3,0),0:(6.4,0)}
for k,(x,y) in P.items():
    ax.add_patch(Circle((x,y),.62,fc='white' if k else '#f3f3f3',ec=C1 if k else GR,lw=2 if k else 1.2,ls='-' if k else '--'))
    ax.text(x,y+.08,f'{k}',ha='center',va='center',fontsize=17,color='k' if k else GR)
    ax.text(x,y-.33,'mod 3',ha='center',va='center',fontsize=7.5,color=GR)
ax.add_patch(FancyArrowPatch((.5,.35),(2.5,.35),arrowstyle='-|>',mutation_scale=12,color=C2,connectionstyle='arc3,rad=-.35'))
ax.add_patch(FancyArrowPatch((2.5,-.35),(.5,-.35),arrowstyle='-|>',mutation_scale=12,color=C2,connectionstyle='arc3,rad=-.35'))
ax.text(1.5,1.02,'II doubles: 1 → 2',ha='center',fontsize=8.5,color=C2); ax.text(1.5,-1.12,'II doubles: 2 → 4 ≡ 1',ha='center',fontsize=8.5,color=C2)
for x in (0,3):
    ax.add_patch(FancyArrowPatch((x-.35,.55),(x+.35,.55),arrowstyle='-|>',mutation_scale=10,color=C3,connectionstyle='arc3,rad=-2.2'))
ax.text(1.5,1.62,'I, III and IV leave the count unchanged mod 3',ha='center',fontsize=8.5,color=C3)
ax.text(6.4,-1.05,'MU has no I\'s:\nnothing gets here',ha='center',va='top',fontsize=8.5,color=GR)
ax.annotate('',(-.65,0),(-1.6,0),arrowprops=dict(arrowstyle='-|>',color='k'))
ax.text(-1.65,.18,'MI',fontsize=9,family='monospace',ha='right')
ax.set_xlim(-2.3,7.4); ax.set_ylim(-1.8,2.0)
fig.tight_layout(); fig.savefig(out+'fig_mod3.png'); plt.close(fig)

# F3: growth of the search, and how long the shortest derivations are
fig,ax=plt.subplots(1,2,figsize=(10,4))
depth,_=bfs(8)
n=np.arange(9); cnt=[sum(1 for v in depth.values() if v<=k) for k in n]
lens=[[len(s) for s,v in depth.items() if v<=k] for k in n]
ax[0].semilogy(n,cnt,'o-',color=C1,label='theorems found')
ax[0].semilogy(n,[max(l) for l in lens],'s--',color=C2,ms=4,label='longest theorem found (symbols)')
ax[0].set_xlabel('steps from MI'); ax[0].set_title('Blind breadth-first search',fontsize=9.5); ax[0].legend(fontsize=8)
S=json.load(open('shortest.json'))
L=np.array([len(k) for k in S]); a=np.array([v[0] for v in S.values()]); c=np.array([v[1] for v in S.values()])
rng=np.random.default_rng(1); j=rng.uniform(-.18,.18,len(L))
ax[1].scatter(L+j-.2,a,s=7,color=C1,alpha=.5,label='shortest derivation')
ax[1].scatter(L+j+.2,c,s=7,color=C2,alpha=.4,label='the recipe in this post')
k='UUUUUUI'; ax[1].annotate('MUUUUUUI: 18 steps',(len(k)-.2,S[k][0]),xytext=(2.0,21),fontsize=8,arrowprops=dict(arrowstyle='->',color='k'))
ax[1].set_xlabel('length of the theorem (symbols after M)'); ax[1].set_ylabel('steps from MI'); ax[1].set_title('Every allowed string up to 8 symbols',fontsize=9.5); ax[1].legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_search.png'); plt.close(fig)

# thumbnail
fig,ax=plt.subplots(figsize=(6,4)); ax.axis('off'); ax.grid(False); fig.patch.set_facecolor('#14243b')
ax.text(.5,.66,'MI  →  …  →  MU ?',ha='center',va='center',fontsize=30,family='monospace',color='white')
ax.text(.5,.34,'#I mod 3:  1  ⇄  2      0',ha='center',va='center',fontsize=16,family='monospace',color='#9fc3ee')
ax.text(.5,.12,'GEB · Part 1',ha='center',fontsize=11,color='#c9a85c')
fig.savefig(out+'thumbnail.png',facecolor=fig.get_facecolor()); plt.close(fig)
print("ok")
