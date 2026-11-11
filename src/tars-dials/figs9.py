import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import binom
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/sci-fi/tars-dials/'
H=lambda p: np.where((p>0)&(p<1),-p*np.log2(np.clip(p,1e-12,1))-(1-p)*np.log2(np.clip(1-p,1e-12,1)),0)
h=np.linspace(0,1,401)
fig,a=plt.subplots(figsize=(6.6,3.6))
a.plot(h*100,1-H(1-h),color=C1,lw=2)
for v,lab,xy in ((0.9,'TARS at 90%: 0.53 bits',(62,0.36)),(0.95,'95%: 0.71 bits',(70,0.88)),(0.5,'50%: answers carry nothing',(30,0.18))):
    c=1-H(1-v); a.plot(v*100,c,'o',color=C2); a.annotate(lab,(v*100,c),xytext=xy,fontsize=8,arrowprops=dict(arrowstyle='->',color=GR))
a.set_xlabel('honesty setting (%)'); a.set_ylabel('information per yes/no answer (bits)'); a.set_ylim(0,1.05)
fig.tight_layout(); fig.savefig(out+'fig_honesty.png'); plt.close(fig)
# repetitions for confidence
def need(p,target):
    for n in range(1,400,2):
        if binom.cdf(n//2,n,p)<=1-target: return n   # wrong majority prob = P(#truthful <= n//2)
    return np.nan
hs=np.linspace(0.55,0.99,45)
fig,a=plt.subplots(figsize=(6.6,3.6))
for tgt,c in ((0.99,C1),(0.999,C3)):
    a.semilogy(hs*100,[need(x,tgt) for x in hs],'o-',ms=3,color=c,label=f'{tgt*100:g}% sure of the majority answer')
a.axvline(90,color=C2,ls=':'); a.text(90.5,60,'TARS',color=C2,fontsize=8.5)
a.set_xlabel('honesty setting (%)'); a.set_ylabel('times to ask the same question'); a.legend(fontsize=8); a.set_xlim(55,100)
fig.tight_layout(); fig.savefig(out+'fig_repeats.png'); plt.close(fig)
for p in (0.9,0.95): print(p,'cap',1-H(1-p),'n99',need(p,0.99),'n999',need(p,0.999))
