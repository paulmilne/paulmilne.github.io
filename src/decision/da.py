"""Decision analysis explainer: AHP (principal eigenvector, consistency, random index),
PCA of an options-by-criteria table, PC1-as-weights pitfall, bootstrap of PCA,
and weight-uncertainty rank acceptability (SMAA-style)."""
import numpy as np, json, itertools, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
out='../../explainers/decision-analysis/'
rng=np.random.default_rng(11)
SAATY=np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9])

def ahp(A, iters=None):
    w,V=np.linalg.eig(A); i=np.argmax(w.real); lam=w[i].real; v=np.abs(V[:,i].real); v/=v.sum()
    n=len(A); CI=(lam-n)/(n-1) if n>2 else 0.0
    return v,lam,CI

# --- random index by simulation ---
RI={}
for n in range(3,10):
    vals=[]
    for _ in range(4000):
        A=np.ones((n,n))
        for i,j in itertools.combinations(range(n),2):
            a=rng.choice(SAATY); A[i,j]=a; A[j,i]=1/a
        vals.append(ahp(A)[2])
    RI[n]=float(np.mean(vals))
print('simulated RI', {k:round(v,3) for k,v in RI.items()})
SAATY_RI={3:.58,4:.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45}

# --- the scenario: 10 data platforms scored on 6 raw criteria (illustrative) ---
opts=list('ABCDEFGHIJ')
raw_names=['Capability','Integration','Security','Low licence cost','Low five-year TCO','Fast deployment']
n=len(opts)
quality=rng.normal(0,1,n); price=.55*quality+rng.normal(0,.85,n); speed=-.35*quality+rng.normal(0,1,n)
cap=7+1.3*quality+rng.normal(0,.5,n); integ=6.5+1.0*quality+rng.normal(0,.8,n); sec=7+.9*quality+rng.normal(0,.9,n)
lic=300+90*price+rng.normal(0,15,n); tco=1500+420*price+rng.normal(0,60,n); dep=9-1.6*speed+rng.normal(0,1,n)
R=np.c_[cap,integ,sec,lic,tco,dep]
R[:,:3]=np.clip(R[:,:3],1,10).round(1); R[:,3]=R[:,3].round(-1); R[:,4]=R[:,4].round(-1); R[:,5]=np.clip(R[:,5],3,18).round()
benefit=np.array([1,1,1,-1,-1,-1])   # higher better / lower better
print('raw table'); [print(o,R[k]) for k,o in enumerate(opts)]
# oriented, standardised scores (higher always better)
Z=(R-R.mean(0))/R.std(0,ddof=1)*benefit
corr=np.corrcoef(Z.T); print('corr\n',corr.round(2))

def pca(Z):
    S=np.cov(Z.T); w,V=np.linalg.eigh(S); o=np.argsort(w)[::-1]; w,V=w[o],V[:,o]
    V=V*np.sign(V[np.argmax(abs(V),0),range(V.shape[1])]); return w,V,Z@V
w,V,S=pca(Z); print('pca eig',w.round(3),'share',(w/w.sum()).round(3))
print('PC1 loadings',dict(zip(raw_names,(V[:,0]*np.sqrt(w[0])).round(2))))
print('PC2 loadings',dict(zip(raw_names,(V[:,1]*np.sqrt(w[1])).round(2))))

# --- AHP over 5 decision criteria (cost criteria merged) ---
crit=['Capability','Integration','Security','Affordability','Speed']
J={(0,1):3,(0,2):2,(0,3):1,(0,4):4,(1,2):1,(1,3):1/3,(1,4):2,(2,3):1/2,(2,4):5,(3,4):4}
def build(J,n=5):
    A=np.ones((n,n))
    for (i,j),a in J.items(): A[i,j]=a; A[j,i]=1/a
    return A
A=build(J); wA,lam,CI=ahp(A); CR=CI/SAATY_RI[5]
gm=np.prod(A,1)**(1/5); gm/=gm.sum()
print('AHP weights',dict(zip(crit,wA.round(3))),'lambda',round(lam,3),'CI',round(CI,3),'CR',round(CR,3)); print('geo-mean',gm.round(3))
# decision scores on 5 criteria: merge the two cost columns
D=np.c_[Z[:,0],Z[:,1],Z[:,2],(Z[:,3]+Z[:,4])/2,Z[:,5]]
# rescale each criterion to 0..1 (min-max) for additive value
Dn=(D-D.min(0))/(D.max(0)-D.min(0))
val=Dn@wA; rankA=np.argsort(-val)
print('AHP values',dict(zip(opts,val.round(3))),'order',''.join(opts[i] for i in rankA))
# PC1-as-weights pitfall: weights from PC1 of the decision table
wd,Vd,Sd=pca((D-D.mean(0))/D.std(0,ddof=1)); pcw=np.abs(Vd[:,0])/np.abs(Vd[:,0]).sum()
print('PC1 weights',dict(zip(crit,pcw.round(3))))
valP=Dn@pcw; rankP=np.argsort(-valP); print('PC1 order',''.join(opts[i] for i in rankP))
# equal weights on the 6 raw criteria: double counting of cost
eq6=np.ones(6)/6; Rn=(Z-Z.min(0))/(Z.max(0)-Z.min(0)); valE=Rn@eq6
eff=np.array([eq6[0],eq6[1],eq6[2],eq6[3]+eq6[4],eq6[5]])
print('equal weights on 6 raw -> effective on 5:',dict(zip(crit,eff.round(3))))

# dominance
dom=[]
for i in range(n):
    for j in range(n):
        if i!=j and np.all(Dn[j]>=Dn[i]) and np.any(Dn[j]>Dn[i]): dom.append((opts[j],opts[i]))
print('dominated pairs (better, worse):',dom)

# --- figures ---
# F1: consistency: eigenvalue vs perturbation, and RI
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
nn=np.array(sorted(RI)); ax[0].plot(nn,[RI[k] for k in nn],'o-',color=C1,label='simulated here (4,000 matrices each)')
ax[0].plot(nn,[SAATY_RI[k] for k in nn],'s--',color=C2,ms=4,label="Saaty's published values")
ax[0].set_xlabel('number of criteria n'); ax[0].set_ylabel('random index RI'); ax[0].legend(fontsize=8); ax[0].set_title('Average CI of random comparison matrices',fontsize=9.5)
# perturb a consistent matrix with log-normal noise
wt=np.array([.35,.15,.3,.12,.08]); sig=np.linspace(0,1.2,25); crs=[]; errs=[]
for s in sig:
    c=[];e=[]
    for _ in range(300):
        Ap=np.ones((5,5))
        for i,j in itertools.combinations(range(5),2):
            a=wt[i]/wt[j]*np.exp(rng.normal(0,s)); Ap[i,j]=a; Ap[j,i]=1/a
        v,l,ci=ahp(Ap); c.append(ci/SAATY_RI[5]); e.append(np.abs(v-wt).max())
    crs.append(np.percentile(c,[10,50,90])); errs.append(np.median(e))
crs=np.array(crs)
ax[1].fill_between(sig,crs[:,0],crs[:,2],color=C1,alpha=.2); ax[1].plot(sig,crs[:,1],color=C1,label='consistency ratio (median, 10–90%)')
ax[1].axhline(.1,color=C2,ls='--',lw=1); ax[1].text(.02,.12,'CR = 0.1',color=C2,fontsize=8)
a2=ax[1].twinx(); a2.plot(sig,errs,color=C3,lw=1.4,ls=':'); a2.set_ylabel('largest weight error',color=C3); a2.spines['top'].set_visible(False); a2.grid(False)
ax[1].set_xlabel('judgement noise σ (log scale: σ = 0.5 means typical errors of a factor of 1.6)'); ax[1].set_ylabel('CR'); ax[1].legend(fontsize=8,loc='upper left')
ax[1].set_title('Noisy judgements raise λmax above n',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_consistency.png'); plt.close(fig)

# F2: correlation heatmap + biplot with decision stick
fig,ax=plt.subplots(1,2,figsize=(11,4.8),gridspec_kw={'width_ratios':[1,1.25]})
im=ax[0].imshow(corr,cmap='RdBu_r',vmin=-1,vmax=1); ax[0].grid(False)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels(raw_names,rotation=40,ha='right',fontsize=8); ax[0].set_yticks(range(6)); ax[0].set_yticklabels(raw_names,fontsize=8)
for i in range(6):
    for j in range(6): ax[0].text(j,i,f'{corr[i,j]:.2f}'.replace('-','−'),ha='center',va='center',fontsize=7.5,color='white' if abs(corr[i,j])>.6 else 'k')
ax[0].set_title('Correlations between criteria\n(oriented so that higher is better)',fontsize=9.5)
L=V[:,:2]*np.sqrt(w[:2]); sc=S[:,:2]/np.abs(S[:,:2]).max(0)*.95
ax[1].scatter(sc[:,0],sc[:,1],s=30,color=C1,zorder=3)
for k,o in enumerate(opts): ax[1].text(sc[k,0]+.04,sc[k,1]+.04,o,fontsize=9,color=C1,weight='bold')
for i,nm in enumerate(raw_names):
    ax[1].annotate('',L[i],(0,0),arrowprops=dict(arrowstyle='-|>',color='#444',lw=1))
    if nm=='Low licence cost': continue
    lab={'Low five-year TCO':'Low licence cost\nand low TCO'}.get(nm,nm)
    p=L[i]*1.13; p={'Security':(L[i,0]-.25,L[i,1]+.02),'Low five-year TCO':(L[i,0]-.05,L[i,1]-.15)}.get(nm,p)
    ax[1].text(*p,lab,fontsize=7.5,ha='center',va='center',bbox=dict(fc='white',ec='none',alpha=.75,pad=.4))
ax[1].text(-1.1,-1.08,'← capable but costly',fontsize=8,color=GR); ax[1].text(1.1,-1.08,'cheaper but basic →',fontsize=8,color=GR,ha='right')
w6=np.array([wA[0],wA[1],wA[2],wA[3]/2,wA[3]/2,wA[4]]); stick=V[:,:2].T@w6; stick=stick/np.linalg.norm(V.T@w6)
ax[1].annotate('',stick,(0,0),arrowprops=dict(arrowstyle='-|>',color=C2,lw=2.6)); ax[1].text(stick[0]+.07,stick[1]+.1,'decision stick',color=C2,fontsize=8,ha='left')
ax[1].set_xlim(-1.15,1.15); ax[1].set_ylim(-1.15,1.15); ax[1].set_aspect('equal')
ax[1].set_xlabel(f'PC1 ({w[0]/w.sum():.0%})'); ax[1].set_ylabel(f'PC2 ({w[1]/w.sum():.0%})')
ax[1].set_title('Options and criteria on the first two components',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_pca.png'); plt.close(fig)
print('stick length (share of weight vector in plane)',np.linalg.norm(V[:,:2].T@w6)/np.linalg.norm(w6))

# F3: weights compared, and rankings
fig,ax=plt.subplots(1,2,figsize=(10.5,3.8))
x=np.arange(5); bw=.27
ax[0].bar(x-bw,wA,bw,color=C2,label='AHP (your priorities)'); ax[0].bar(x,pcw,bw,color=C1,label='PC1 loadings, normalised'); ax[0].bar(x+bw,eff,bw,color=GR,label='"equal" weights on 6 raw criteria')
ax[0].set_xticks(x); ax[0].set_xticklabels(crit,fontsize=8); ax[0].set_ylabel('weight'); ax[0].set_ylim(0,.48); ax[0].legend(fontsize=7.5,loc='upper center',ncol=1); ax[0].set_title('Three sets of weights',fontsize=9.5)
for k,(vals,c,lab) in enumerate(((val,C2,'AHP'),(valP,C1,'PC1'),(Dn@eff,GR,'equal (raw)'))):
    order=np.argsort(-vals); rk=np.empty(n,int); rk[order]=np.arange(1,n+1)
    ax[1].plot([k]*n,rk,'o',color=c)
    for i,o in enumerate(opts): ax[1].text(k+.08,rk[i],o,fontsize=8,va='center')
    if k<2:
        nxt=[val,valP,Dn@eff][k+1]; o2=np.argsort(-nxt); rk2=np.empty(n,int); rk2[o2]=np.arange(1,n+1)
        for i in range(n): ax[1].plot([k,k+1],[rk[i],rk2[i]],color='#bbb',lw=.8,zorder=0)
ax[1].set_xticks([0,1,2]); ax[1].set_xticklabels(['AHP weights','PC1 as weights','equal on raw'],fontsize=8.5); ax[1].invert_yaxis(); ax[1].set_ylabel('rank'); ax[1].set_title('The ranking depends on where the weights come from',fontsize=9.5); ax[1].grid(False)
fig.tight_layout(); fig.savefig(out+'fig_weights.png'); plt.close(fig)

# F4: bootstrap of PC1 loadings; F5: rank acceptability under weight uncertainty
fig,ax=plt.subplots(1,2,figsize=(11,4))
B=2000; boots=[]
for _ in range(B):
    idx=rng.integers(0,n,n)
    if len(set(idx))<4: continue
    Zb=Z[idx]; Zb=(Zb-Zb.mean(0))/(Zb.std(0,ddof=1)+1e-9); wb,Vb,_=pca(Zb); v=Vb[:,0]
    v=v*np.sign(v@V[:,0]); boots.append(v)
boots=np.array(boots)
ax[0].boxplot([boots[:,i] for i in range(6)],tick_labels=[s.replace(' ','\n') for s in raw_names],showfliers=False)
ax[0].plot(range(1,7),V[:,0],'o',color=C2,label='full sample'); ax[0].axhline(0,color='k',lw=.6)
ax[0].set_ylabel('PC1 coefficient'); ax[0].legend(fontsize=8); ax[0].set_title('PC1 from bootstrap resamples of the 10 options',fontsize=9.5); ax[0].tick_params(axis='x',labelsize=7.5)
angle=np.degrees(np.arccos(np.clip(np.abs(boots@V[:,0]),0,1))); print('bootstrap PC1 angle median/90%', np.median(angle).round(1), np.percentile(angle,90).round(1))
M=20000; Wd=rng.dirichlet(wA*30,M); ranks=np.zeros((n,n))
for wv in Wd:
    o=np.argsort(-(Dn@wv)); ranks[o,np.arange(n)]+=1
ranks/=M
ordr=np.argsort(-ranks[:,0]-.5*ranks[:,1]-.25*ranks[:,2])
im=ax[1].imshow(ranks[rankA][:, :n],cmap='Blues',vmin=0,vmax=1,aspect='auto'); ax[1].grid(False)
ax[1].set_yticks(range(n)); ax[1].set_yticklabels([opts[i] for i in rankA]); ax[1].set_xticks(range(n)); ax[1].set_xticklabels(range(1,n+1))
for i in range(n):
    for j in range(n):
        p=ranks[rankA[i],j]
        if p>=.03: ax[1].text(j,i,f'{p:.0%}',ha='center',va='center',fontsize=7,color='white' if p>.5 else 'k')
ax[1].set_xlabel('rank'); ax[1].set_ylabel('option (in AHP order)'); ax[1].set_title('Rank probabilities when the AHP weights are uncertain',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_robust.png'); plt.close(fig)
print('P(rank1)',dict(zip(opts,ranks[:,0].round(3))))
cvw=Wd.std(0)/Wd.mean(0); print('weight CV',cvw.round(2))

# thumbnail
fig,ax=plt.subplots(figsize=(6,4)); ax.bar(range(5),wA,color=[C1,C3,C2,BR,PU]); ax.set_xticks(range(5)); ax.set_xticklabels(crit,fontsize=9)
ax.set_title(f'AHP weights: principal eigenvector, λmax = {lam:.2f}',fontsize=11); fig.tight_layout(); fig.savefig(out+'thumbnail.png'); plt.close(fig)

json.dump({'opts':opts,'crit':crit,'Dn':Dn.round(4).tolist(),'J':{f'{i},{j}':a for (i,j),a in J.items()},'RI':SAATY_RI,
           'raw_names':raw_names,'R':R.tolist()},open(out+'scenario.json','w'))
