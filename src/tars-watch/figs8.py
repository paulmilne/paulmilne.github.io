import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from morse import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/sci-fi/tars-watch/'
C=capacity()
# F1 STAY timing diagram
seg=encode('STAY'); t=0
fig,a=plt.subplots(figsize=(10,1.9))
for on,d in seg:
    if on: a.add_patch(plt.Rectangle((t,0),d,1,color=BR))
    t+=d
x=0
for ch in 'STAY':
    L=units(M[ch]); a.text(x+L/2,1.25,f'{ch}  {M[ch]}',ha='center',fontsize=10); x+=L+3
a.set_xlim(-0.5,t+0.5); a.set_ylim(-0.3,1.7); a.set_yticks([]); a.set_xlabel('time (units)'); a.grid(False); a.spines['left'].set_visible(False)
fig.tight_layout(); fig.savefig(out+'fig_stay.png'); plt.close(fig)
# F2 counting: number of valid sequences of duration T
T=60; N0=np.zeros(T+1); N1=np.zeros(T+1); N0[0]=1   # N0: ends after element (or start), N1: ends after a space
for t in range(1,T+1):
    for d in (2,4):
        if t>=d: N0[t]+=N0[t-d]+N1[t-d]
    for d in (3,6):
        if t>=d: N1[t]+=N0[t-d]
N=N0+N1
fig,a=plt.subplots(figsize=(6.6,3.6))
tt=np.arange(T+1); a.plot(tt[1:],np.log2(np.maximum(N[1:],1)),'o',ms=3,color=C1,label='valid Morse sequences lasting T units (log₂ count)')
a.plot(tt,C*tt-1,color=C2,lw=1.5,label=f'slope = capacity = {C:.3f} bits per unit')
a.set_xlabel('duration T (units)'); a.set_ylabel('bits'); a.legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_capacity.png'); plt.close(fig)
# F3 time to send
sizes=np.logspace(2,10,200)    # bits
fig,a=plt.subplots(figsize=(7,4))
for tau,c,lab in ((0.15,C1,'0.15 s per unit (8 words a minute, a practised reader)'),(1.0,C2,'1 s per unit (one tick a second)')):
    a.loglog(sizes/8,sizes/(C/tau)/3600,color=c,lw=2,label=lab)
for b,lab in ((280*8,'a tweet\n(280 chars)'),(3000*8,'a page of\nequations'),(1e6*8,'a megabyte'),(1e9*8,'a gigabyte')):
    a.axvline(b/8,color=GR,ls=':',lw=.8); a.text(b/8*1.15,2e-3,lab,fontsize=7.5,color=GR)
for h,lab in ((4,'an evening'),(24*365,'a year'),(24*365*80,'a lifetime')):
    a.axhline(h,color='k',lw=.6,ls='--'); a.text(15,h*1.25,lab,fontsize=8)
a.set_xlabel('message size (bytes)'); a.set_ylabel('hours at Shannon capacity'); a.legend(fontsize=7.5,loc='upper left',bbox_to_anchor=(0.1,0.98)); a.set_xlim(12,1.5e9); a.set_ylim(1e-3,2e7)
fig.tight_layout(); fig.savefig(out+'fig_time.png'); plt.close(fig)
# F4 jitter: character error rate by edit distance
def lev(a,b):
    d=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        p,d[0]=d[0],i
        for j,cb in enumerate(b,1):
            p,d[j]=d[j],min(d[j]+1,d[j-1]+1,p+(ca!=cb))
    return d[-1]
rng=np.random.default_rng(0); alph=list(M.keys()); sig=np.linspace(0,0.6,25)
def cer(s,n=40):
    e=t=0
    for _ in range(n):
        msg=' '.join(''.join(rng.choice(alph,5)) for _ in range(12))
        dec=decode(noisy(encode(msg),s,rng)); e+=lev(msg,dec); t+=len(msg)
    return e/t
fig,a=plt.subplots(figsize=(6.6,3.6))
a.semilogy(sig,[max(cer(s),1e-4) for s in sig],'o-',color=C2)
a.set_xlabel('timing jitter σ (in units, for a one-unit element)'); a.set_ylabel('character error rate'); a.set_ylim(1e-4,1.5)
a.axvspan(0,0.12,color=C3,alpha=.08); a.text(0.01,2e-4,'reads perfectly',fontsize=8,color=C3)
fig.tight_layout(); fig.savefig(out+'fig_jitter.png'); plt.close(fig)
