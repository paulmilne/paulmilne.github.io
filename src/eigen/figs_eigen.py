"""Figures for the eigenvectors explainer."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='../../explainers/eigenvectors/'
th=np.linspace(0,2*np.pi,400); circ=np.vstack([np.cos(th),np.sin(th)])

def panel(ax,A,title,lim=3.2,grid=True,vecs=True):
    A=np.array(A,float)
    if grid:
        for k in np.arange(-3,3.01,.5):
            for P in (np.array([[k,k],[-3,3]]),np.array([[-3,3],[k,k]])):
                Q=A@P; ax.plot(Q[0],Q[1],color='#c9d6e8',lw=.6,zorder=0)
    ax.plot(circ[0],circ[1],color=GR,lw=1,ls='--')
    E=A@circ; ax.plot(E[0],E[1],color='k',lw=1.6)
    # sample vectors and images
    for t in np.linspace(0,np.pi,7)[:-1]+.2:
        v=np.array([np.cos(t),np.sin(t)]); w=A@v
        ax.annotate('',w,v,arrowprops=dict(arrowstyle='->',color='#bbbbbb',lw=.8))
    lam,V=np.linalg.eig(A)
    txt=[]
    if vecs and np.all(np.abs(lam.imag)<1e-9):
        if abs(np.linalg.det(V))<1e-6: V=V[:,:1]; lam=lam[:1]
        for i,(l,c) in enumerate(zip(lam.real,(C2,C3))):
            v=V[:,i].real; v/=np.linalg.norm(v)
            ax.plot([-4*v[0],4*v[0]],[-4*v[1],4*v[1]],color=c,lw=1,alpha=.5)
            ax.annotate('',l*v,(0,0),arrowprops=dict(arrowstyle='-|>',color=c,lw=2.2))
            ax.annotate('',v,(0,0),arrowprops=dict(arrowstyle='-|>',color=c,lw=1,ls='--'))
            txt.append(f'λ = {0 if abs(l)<1e-9 else l:.3g}'.replace('-','−'))
        if V.shape[1]==1: txt=[f'λ = {lam[0].real:.3g} (twice), one eigenvector']
    else:
        r=abs(lam[0]); txt=[f'λ = {lam[0].real:.2g} ± {abs(lam[0].imag):.2g}i', f'|λ| = {r:.2g}, angle {np.degrees(abs(np.angle(lam[0]))):.0f}°']
    ax.set_title(title+'\n'+'  ·  '.join(txt),fontsize=9)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.tick_params(labelsize=7)

# F1: the example matrix
fig,ax=plt.subplots(1,1,figsize=(5.6,5.6))
panel(ax,[[2,1],[1,2]],r'$A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$' if False else 'A = [[2, 1], [1, 2]]')
fig.tight_layout(); fig.savefig(out+'fig_example.png'); plt.close(fig)

# F2: a gallery of 2x2 matrices
cases=[([[2,1],[1,2]],'Symmetric'),([[1.5,1],[0,.5]],'Not symmetric'),([[1,1],[0,1]],'Shear'),
       ([[0.8,-1.2],[1.2,0.8]],'Rotation and scaling'),([[1,0],[0,-1]],'Reflection'),([[.5,.5],[.5,.5]],'Projection')]
fig,axs=plt.subplots(2,3,figsize=(11,7.6))
for ax,(A,t) in zip(axs.flat,cases): panel(ax,A,t,lim=2.6)
fig.tight_layout(); fig.savefig(out+'fig_gallery.png'); plt.close(fig)

# F3: repeated application: Fibonacci and power iteration
F=np.array([[1,1],[1,0]]); phi=(1+5**.5)/2
fig,ax=plt.subplots(1,2,figsize=(10,4))
x=np.array([1.,0.]); xs=[x]
for k in range(10): x=F@x; xs.append(x)
xs=np.array(xs); 
lam,V=np.linalg.eig(F)
for i,c in zip(range(2),(C2,C3)):
    v=V[:,i]; ax[0].plot([-12*v[0],12*v[0]],[-12*v[1],12*v[1]],color=c,lw=1,alpha=.6,label=f'eigenvector, λ = {lam[i].real:.3f}'.replace('-','−'))
ax[0].plot(xs[:7,0],xs[:7,1],'o-',color=C1,ms=4)
for k,(a,b) in enumerate(xs[:7]): ax[0].annotate(f'({a:.0f}, {b:.0f})',(a,b),fontsize=7,xytext=(4,-10),textcoords='offset points')
ax[0].set_xlim(-2,15); ax[0].set_ylim(-2,10); ax[0].set_aspect('equal'); ax[0].legend(fontsize=8,loc='upper left')
ax[0].set_title('Fibonacci pairs (Fₙ₊₁, Fₙ) under A = [[1, 1], [1, 0]]',fontsize=9.5)
fib=[1,1]; 
for _ in range(20): fib.append(fib[-1]+fib[-2])
r=np.array(fib[1:])/np.array(fib[:-1]); n=np.arange(1,len(r)+1)
ax[1].semilogy(n,np.abs(r-phi),'o-',color=C1,ms=3,label='|Fₙ₊₁/Fₙ − φ|')
ax[1].semilogy(n,2.5*(1/phi**2)**n,'--',color=C2,label='∝ |λ₂/λ₁|ⁿ = (1/φ²)ⁿ')
ax[1].set_xlabel('n'); ax[1].legend(fontsize=8); ax[1].set_title('Ratio of successive terms → φ = 1.618…',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_fibonacci.png'); plt.close(fig)

# F4: Markov chain: stationary distribution and mixing
P=np.array([[.7,.2,.1],[.3,.5,.2],[.2,.3,.5]])   # rows: from sunny, cloudy, rainy
w,VL=np.linalg.eig(P.T); i=np.argmin(abs(w-1)); pi=np.real(VL[:,i]); pi/=pi.sum()
lam2=sorted(abs(w))[-2]
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for start,ls in ((np.array([1,0,0.]),'-'),(np.array([0,0,1.]),'--')):
    d=[start]
    for k in range(15): d.append(d[-1]@P)
    d=np.array(d)
    for j,(c,lab) in enumerate(zip((BR,GR,C1),('sunny','cloudy','rainy'))):
        ax[0].plot(d[:,j],ls,color=c,label=lab if ls=='-' else None)
for j,c in enumerate((BR,GR,C1)): ax[0].axhline(pi[j],color=c,lw=.6,ls=':')
ax[0].set_xlabel('day'); ax[0].set_ylabel('probability'); ax[0].legend(fontsize=8,loc='center right')
ax[0].set_title('Starting sunny (solid) or rainy (dashed)',fontsize=9.5)
d=[np.array([1,0,0.])]
for k in range(25): d.append(d[-1]@P)
err=[0.5*abs(x-pi).sum() for x in d]
ax[1].semilogy(err,'o-',color=C1,ms=3,label='distance from stationary')
ax[1].semilogy(np.arange(26),err[1]/lam2*lam2**np.arange(26),'--',color=C2,label=f'∝ |λ₂|ᵏ, |λ₂| = {lam2:.3f}')
ax[1].set_xlabel('day'); ax[1].legend(fontsize=8); ax[1].set_title('Convergence is set by the second eigenvalue',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_markov.png'); plt.close(fig)
print('pi',pi.round(4),'eig',np.round(w,4))

# F5: trace-determinant plane with phase portraits
fig=plt.figure(figsize=(10.5,6.2))
ax=fig.add_axes([.07,.1,.5,.82])
T=np.linspace(-4,4,400); ax.plot(T,T**2/4,color='k',lw=1); ax.axhline(0,color='k',lw=1); ax.plot([0,0],[0,5],color='k',lw=1)
ax.fill_between(T,T**2/4,5,where=T<0,color=C3,alpha=.08); ax.fill_between(T,T**2/4,5,where=T>0,color=C2,alpha=.08)
ax.fill_between(T,-3,0,color=GR,alpha=.08)
lab=[(-2.9,.6,'stable node'),(1.8,.6,'unstable node'),(-1.4,3.4,'stable spiral'),(.8,3.4,'unstable spiral'),(0,-1.8,'saddle'),(.12,4.55,'centres (T = 0)')]
for x,y,s in lab: ax.text(x,y,s,ha='left' if 'centres' in s else 'center',fontsize=9)
ax.text(2.9,2.4,'T² = 4D',fontsize=8.5,rotation=50)
ax.set_xlim(-4,4); ax.set_ylim(-3,5); ax.set_xlabel('trace T = λ₁ + λ₂'); ax.set_ylabel('determinant D = λ₁λ₂')
ax.set_title('Linear system ẋ = Ax: the eigenvalues decide everything',fontsize=9.5)
ex=[([[-2,0],[0,-1]],'stable node',(-3,2)),([[-.3,-2],[2,-.3]],'stable spiral',(-.6,4.09)),([[0,-1.5],[1.5,0]],'centre',(0,2.25)),([[1,0],[0,-1]],'saddle',(0,-1)),([[.3,-2],[2,.3]],'unstable spiral',(.6,4.09)),([[1.5,0],[0,1]],'unstable node',(2.5,1.5))]
g=np.linspace(-1,1,15); X,Y=np.meshgrid(g,g)
for k,(A,name,pt) in enumerate(ex):
    A=np.array(A,float); ax.plot(*pt,'o',color='k',ms=4)
    a2=fig.add_axes([.62+.13*(k%3),.56-.42*(k//3),.12,.34]); a2.set_aspect('equal')
    U=A[0,0]*X+A[0,1]*Y; V_=A[1,0]*X+A[1,1]*Y
    a2.streamplot(X,Y,U,V_,density=.7,color=C1,linewidth=.6,arrowsize=.6)
    a2.set_xticks([]); a2.set_yticks([]); a2.set_title(name,fontsize=8); a2.grid(False)
fig.savefig(out+'fig_stability.png'); plt.close(fig)

# F6: two masses, three springs
fig,ax=plt.subplots(1,2,figsize=(10,3.4))
K=np.array([[2,-1],[-1,2]]); w2,M=np.linalg.eigh(K)
t=np.linspace(0,20,800)
for i,(c,name) in enumerate(zip((C1,C2),('in phase','in opposition'))):
    m=M[:,i]*np.sign(M[0,i]); om=np.sqrt(w2[i])
    ax[0].plot(t,m[0]*np.cos(om*t)+0,color=c,lw=1.2,label=f'mode {i+1}: {name}, ω = {om:.3f}')
    ax[0].plot(t,m[1]*np.cos(om*t)+(2.2 if i==0 else -2.2) if False else m[1]*np.cos(om*t),color=c,lw=1.2,ls='--')
ax[0].set_xlabel('time'); ax[0].set_ylabel('displacement'); ax[0].legend(fontsize=7.5,loc='lower left'); ax[0].set_ylim(-1.4,1.2)
ax[0].set_title('Each mode alone: mass 1 solid, mass 2 dashed',fontsize=9.5)
x0=np.array([1,0.]); c=M.T@x0
X=np.array([sum(c[i]*M[:,i]*np.cos(np.sqrt(w2[i])*tt) for i in range(2)) for tt in t])
ax[1].plot(t,X[:,0],color=C1,lw=1,label='mass 1'); ax[1].plot(t,X[:,1],color=C2,lw=1,label='mass 2')
ax[1].set_xlabel('time'); ax[1].legend(fontsize=8,loc='lower left'); ax[1].set_title('Pull mass 1 and let go: the two modes beat',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_modes.png'); plt.close(fig)

# F7: covariance ellipse
rng=np.random.default_rng(3); S=np.array([[3,1.6],[1.6,1.4]]); Xs=rng.multivariate_normal([0,0],S,400)
C=np.cov(Xs.T); w,V=np.linalg.eigh(C)
fig,ax=plt.subplots(figsize=(5.4,4.6))
ax.scatter(Xs[:,0],Xs[:,1],s=6,color=C1,alpha=.4)
for i,c in zip((1,0),(C2,C3)):
    u=V[:,i]*(1 if V[0,i]>0 else -1); v=u*np.sqrt(w[i])*2; ax.annotate('',v,(0,0),arrowprops=dict(arrowstyle='-|>',color=c,lw=2.4))
    ax.text(*(v+u*.55),f'λ = {w[i]:.2f}',color=c,fontsize=9,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=1))
ang=np.degrees(np.arctan2(V[1,1],V[0,1]))
for k in (1,2): ax.add_patch(Ellipse((0,0),2*k*np.sqrt(w[1]),2*k*np.sqrt(w[0]),angle=ang,fill=False,color='k',lw=.8,ls='--' if k==2 else '-'))
ax.set_aspect('equal'); ax.set_xlim(-6,6); ax.set_ylim(-4.5,4.5)
ax.set_title('Covariance eigenvectors are the axes of the cloud\n(arrows drawn 2 standard deviations long)',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_covariance.png'); plt.close(fig)
print('cov',C.round(3),'eig',w.round(3))

# thumbnail
fig,ax=plt.subplots(figsize=(6,4)); panel(ax,[[2,1],[1,2]],'',lim=3.3); ax.set_title(''); ax.axis('off')
fig.savefig(out+'thumbnail.png',bbox_inches='tight'); plt.close(fig)
