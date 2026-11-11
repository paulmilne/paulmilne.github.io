import numpy as np, json
from scipy.optimize import fsolve
g=0.5
def F(x,K,w0):
    z1=x[0]+1j*x[1]; z2=x[2]+1j*x[3]; z=(z1+z2)/2
    d1=(1j*w0-g)*z1+K/2*(z-np.conj(z)*z1**2); d2=(-1j*w0-g)*z2+K/2*(z-np.conj(z)*z2**2)
    return np.array([d1.real,d1.imag,d2.real,d2.imag])
def jac(x,K,w0,h=1e-7):
    J=np.zeros((4,4)); f0=F(x,K,w0)
    for i in range(4):
        e=np.zeros(4); e[i]=h; J[:,i]=(F(x+e,K,w0)-f0)/h
    return J
def branch(w0):
    # symmetric: z1=rho e^{i phi}, z2=conj; unknowns rho,phi,K at fixed r
    out=[]; guess=None
    for r in np.linspace(0.002,0.97,500):
        def eq(u):
            rho,phi,K=u; z1=rho*np.exp(1j*phi)
            d=(1j*w0-g)*z1+K/2*(r-r*z1**2); return [d.real,d.imag,rho*np.cos(phi)-r]
        if guess is None:
            K0=2*(g*g+w0*w0)/g; guess=[r/np.cos(np.arctan(w0/g)) if True else r, -np.arctan(w0/g)*0.5,K0]
            # solve small r: try several phi
            best=None
            for phi0 in np.linspace(-1.5,1.5,31):
                u,info,ier,msg=fsolve(eq,[r/np.cos(phi0),phi0,K0],full_output=True)
                if ier==1 and u[0]<=1 and u[2]>0: best=u; break
            guess=best
        u,info,ier,msg=fsolve(eq,guess,full_output=True)
        if ier!=1: continue
        guess=u; rho,phi,K=u
        x=np.array([rho*np.cos(phi),rho*np.sin(phi),rho*np.cos(phi),-rho*np.sin(phi)])
        ev=np.linalg.eigvals(jac(x,K,w0)); stable=bool(np.max(ev.real)<1e-6)
        out.append((float(r),float(K),stable))
    return out
res={str(w0):branch(w0) for w0 in (0.2,0.4,1.0)}
json.dump(res,open('branches.json','w'))
for k,v in res.items():
    A=np.array(v); print(k,'Kmin',A[:,1].min(),'at r',A[A[:,1].argmin(),0],'K at r->0',A[0,1], 'stable frac',A[:,2].mean())
