import numpy as np, collections
g=0.5
W=np.linspace(0,2.2,111)*g; Kg=np.linspace(0,8,121)*g
w0,K=np.meshgrid(W,Kg)
def run(z1,z2,T=900,dt=0.01):
    n=int(T/dt); e=np.zeros(w0.shape); l=np.zeros(w0.shape); lmin=np.full(w0.shape,9.)
    for k in range(n):
        z=(z1+z2)/2
        d1=(1j*w0-g)*z1+K/2*(z-np.conj(z)*z1**2); d2=(-1j*w0-g)*z2+K/2*(z-np.conj(z)*z2**2)
        z1=z1+dt*d1; z2=z2+dt*d2
        if k>n*0.7:
            r=abs((z1+z2)/2)
            if k<n*0.85: e=np.maximum(e,r)
            else: l=np.maximum(l,r); lmin=np.minimum(lmin,r)
    inc=(l<1e-3)|(l<0.97*e)
    return np.where(inc,0,np.where(l-lmin>0.01,2,1))
lo=run(np.full(w0.shape,0.01+0.003j),np.full(w0.shape,0.008-0.002j))
hi=run(np.full(w0.shape,0.95+0j),np.full(w0.shape,0.95+0j))
np.savez('phase.npz',W=W/g,K=Kg/g,lo=lo,hi=hi)
print(collections.Counter(zip(lo.ravel(),hi.ravel())))
