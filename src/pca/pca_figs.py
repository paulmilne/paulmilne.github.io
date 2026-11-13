"""PCA explainer: computations and figures. PCA is done by hand (eigh of the
covariance matrix) and checked against the SVD."""
import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'; CL=[C1,C2,C3]
out='../../explainers/pca/'
rng=np.random.default_rng(7)

def pca(X, scale=False):
    Xc=X-X.mean(0)
    if scale: Xc=Xc/X.std(0,ddof=1)
    S=np.cov(Xc.T); w,V=np.linalg.eigh(S); o=np.argsort(w)[::-1]; w,V=w[o],V[:,o]
    V=V*np.sign(V[np.argmax(abs(V),0),range(V.shape[1])])   # sign convention: largest loading positive
    # check against SVD
    U,s,Vt=np.linalg.svd(Xc,full_matrices=False)
    assert np.allclose(s**2/(len(X)-1),w[:len(s)])
    return w,V,Xc@V,Xc

# F1: PCA line vs regression lines
x=rng.normal(0,1,120); y=.6*x+rng.normal(0,.6,120); X=np.c_[x,y]
w,V,Z,Xc=pca(X)
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
for a in ax: a.scatter(Xc[:,0],Xc[:,1],s=9,color=C1,alpha=.55); a.set_aspect('equal'); a.set_xlim(-3.2,3.2); a.set_ylim(-2.6,2.6)
v=V[:,0]; t=np.linspace(-4,4,2)
P=np.outer(Z[:,0],v)
for p,q in zip(Xc,P): ax[0].plot([p[0],q[0]],[p[1],q[1]],color=C2,lw=.6,alpha=.6)
ax[0].plot(t*v[0],t*v[1],color=C2,lw=2,label='first principal component')
ax[0].set_title('PCA minimises perpendicular distances',fontsize=9.5); ax[0].legend(fontsize=8,loc='upper left')
b1=np.polyfit(Xc[:,0],Xc[:,1],1)[0]; b2=1/np.polyfit(Xc[:,1],Xc[:,0],1)[0]
ax[1].plot(t,b1*t,color=C3,lw=1.6,label=f'regress y on x (slope {b1:.2f})')
ax[1].plot(t,b2*t,color=BR,lw=1.6,label=f'regress x on y (slope {b2:.2f})')
ax[1].plot(t*v[0],t*v[1],color=C2,lw=2,label=f'PC1 (slope {v[1]/v[0]:.2f})')
ax[1].set_title('It treats x and y symmetrically; regression does not',fontsize=9.5); ax[1].legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_lines.png'); plt.close(fig)

# Wine
W=load_wine(); Xw=W.data; yw=W.target; names=[n.replace('od280/od315_of_diluted_wines','OD280/OD315').replace('_',' ').replace('color','colour') for n in W.feature_names]
wr,Vr,Zr,_=pca(Xw,False); ws,Vs,Zs,Xs=pca(Xw,True)
print('raw var share PC1',wr[0]/wr.sum(), 'top loading', names[np.argmax(abs(Vr[:,0]))], Vr[np.argmax(abs(Vr[:,0])),0])
print('std eig',ws.round(3),'share',(ws/ws.sum()).round(3),'cum',np.cumsum(ws/ws.sum()).round(3))
fig,ax=plt.subplots(1,2,figsize=(10,4.3))
for k,c in enumerate(CL):
    ax[0].scatter(Zr[yw==k,0],Zr[yw==k,1],s=10,color=c,alpha=.7,label=f'cultivar {k+1}')
    ax[1].scatter(Zs[yw==k,0],Zs[yw==k,1],s=10,color=c,alpha=.7,label=f'cultivar {k+1}')
ax[0].set_title(f'Raw units: PC1 is {wr[0]/wr.sum():.1%} of variance, and almost all proline',fontsize=9)
ax[1].set_title(f'Standardised: PC1 and PC2 hold {ws[:2].sum()/ws.sum():.0%} of variance',fontsize=9)
for a in ax: a.set_xlabel('PC1 score'); a.set_ylabel('PC2 score'); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out+'fig_scaling.png'); plt.close(fig)

# scree with parallel analysis
B=200; null=np.zeros((B,13))
for b in range(B):
    R=rng.normal(size=Xw.shape); null[b]=pca(R,True)[0]
pa=np.percentile(null,95,0)
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
k=np.arange(1,14)
ax[0].plot(k,ws,'o-',color=C1,label='wine (standardised)'); ax[0].plot(k,pa,'s--',color=GR,ms=4,label='random data, 95th percentile')
ax[0].axhline(1,color=BR,lw=.8,ls=':'); ax[0].text(12.8,1.08,'λ = 1',fontsize=8,color=BR,ha='right')
ax[0].set_xlabel('component'); ax[0].set_ylabel('eigenvalue (variance)'); ax[0].legend(fontsize=8); ax[0].set_title('Scree plot with parallel analysis',fontsize=9.5)
ax[1].bar(k,ws/ws.sum(),color=C1,alpha=.7,label='each component'); ax[1].step(k,np.cumsum(ws/ws.sum()),where='mid',color=C2,label='cumulative')
ax[1].axhline(.8,color=GR,lw=.8,ls=':'); ax[1].set_xlabel('component'); ax[1].set_ylabel('share of variance'); ax[1].legend(fontsize=8,loc='center right'); ax[1].set_title('Variance explained',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_scree.png'); plt.close(fig)
print('parallel', pa.round(3)[:4], 'n retained', int((ws>pa).sum()))

# biplot
fig,ax=plt.subplots(figsize=(7.4,6.2))
sc=Zs[:,:2]/np.abs(Zs[:,:2]).max(0)
for kk,c in enumerate(CL): ax.scatter(sc[yw==kk,0],sc[yw==kk,1],s=9,color=c,alpha=.45,label=f'cultivar {kk+1}')
L=Vs[:,:2]*np.sqrt(ws[:2])   # loadings = correlations of variables with PC scores
for i,n in enumerate(names):
    ax.annotate('',L[i],(0,0),arrowprops=dict(arrowstyle='-|>',color='#333',lw=1))
    pos={'total phenols':(L[i,0]+.05,L[i,1]+.07),'proanthocyanins':(L[i,0]-.02,L[i,1]-.07),'flavanoids':(L[i,0]+.13,L[i,1]),
         'nonflavanoid phenols':(L[i,0]-.12,L[i,1]+.06),'alcalinity of ash':(L[i,0]-.1,L[i,1]-.06)}.get(n,tuple(L[i]*1.1))
    ax.text(*pos,n,fontsize=7.5,ha='center',va='center',bbox=dict(fc='white',ec='none',alpha=.75,pad=.5))
ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_aspect('equal'); ax.legend(fontsize=8,loc='lower left')
ax.set_xlabel(f'PC1 ({ws[0]/ws.sum():.0%})'); ax.set_ylabel(f'PC2 ({ws[1]/ws.sum():.0%})')
ax.set_title('Biplot: wines (dots, rescaled) and variables (arrows: correlation with each PC)',fontsize=9)
fig.tight_layout(); fig.savefig(out+'fig_biplot.png'); plt.close(fig)
print('loadings PC1', dict(zip(names,L[:,0].round(2)))); print('loadings PC2', dict(zip(names,L[:,1].round(2))))

# pitfall: largest variance is not the useful direction
n=150; a=np.r_[rng.normal(0,2.2,n),rng.normal(0,2.2,n)]; bb=np.r_[rng.normal(-.7,.3,n),rng.normal(.7,.3,n)]
th=np.radians(25); Rm=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); Xp=(Rm@np.vstack([a,bb])).T; yp=np.r_[np.zeros(n),np.ones(n)]
wp,Vp,Zp,Xpc=pca(Xp)
fig,ax=plt.subplots(1,3,figsize=(11,3.5),gridspec_kw={'width_ratios':[1.4,1,1]})
for kk,c in enumerate((C1,C2)): ax[0].scatter(Xpc[yp==kk,0],Xpc[yp==kk,1],s=7,color=c,alpha=.5,label=f'group {"AB"[kk]}')
for i,(c,lab) in enumerate(((BR,'PC1'),(C3,'PC2'))):
    v=Vp[:,i]*np.sqrt(wp[i])*2; ax[0].annotate('',v,(0,0),arrowprops=dict(arrowstyle='-|>',color=c,lw=2.4)); ax[0].text(*(v*1.15),lab,color=c,fontsize=9)
ax[0].set_aspect('equal'); ax[0].legend(fontsize=8,loc='lower right'); ax[0].set_title('Two groups, separated along the narrow direction',fontsize=9)
for i,(c,lab) in enumerate(((BR,'PC1'),(C3,'PC2'))):
    for kk,cc in enumerate((C1,C2)): ax[i+1].hist(Zp[yp==kk,i],bins=30,color=cc,alpha=.6)
    ax[i+1].set_title(f'Scores on {lab}: {wp[i]/wp.sum():.0%} of variance',fontsize=9); ax[i+1].set_yticks([])
fig.tight_layout(); fig.savefig(out+'fig_pitfall.png'); plt.close(fig)

# digits: eigen-digits and reconstruction
D=load_digits(); Xd=D.data; wd,Vd,Zd,Xdc=pca(Xd); mu=Xd.mean(0)
cum=np.cumsum(wd)/wd.sum(); print('digits k for 80/90/95%', [int(np.searchsorted(cum,q)+1) for q in (.8,.9,.95)])
fig,ax=plt.subplots(2,8,figsize=(10,3.1))
ax[0,0].imshow(mu.reshape(8,8),cmap='gray_r'); ax[0,0].set_title('mean',fontsize=8)
for j in range(7):
    ax[0,j+1].imshow(Vd[:,j].reshape(8,8),cmap='RdBu_r',vmin=-.35,vmax=.35); ax[0,j+1].set_title(f'PC{j+1} ({wd[j]/wd.sum():.0%})',fontsize=8)
idx=5; x0=Xd[idx]
for j,kk in enumerate([1,2,5,10,15,20,30,64]):
    rec=mu+Vd[:,:kk]@(Vd[:,:kk].T@(x0-mu))
    ax[1,j].imshow(rec.reshape(8,8),cmap='gray_r',vmin=0,vmax=16); ax[1,j].set_title(f'{kk} PC{"s" if kk>1 else ""} ({cum[kk-1]:.0%})',fontsize=8)
for a in ax.flat: a.set_xticks([]); a.set_yticks([]); a.grid(False)
fig.tight_layout(); fig.savefig(out+'fig_digits.png'); plt.close(fig)
print('digit label', D.target[idx])

# thumbnail
fig,ax=plt.subplots(figsize=(6,4))
for kk,c in enumerate(CL): ax.scatter(Zs[yw==kk,0],Zs[yw==kk,1],s=14,color=c,alpha=.7)
ax.axis('off'); fig.savefig(out+'thumbnail.png',bbox_inches='tight'); plt.close(fig)
