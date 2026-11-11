import numpy as np, sys, json
from bridge import *
def run(N,T=1500,dt=0.004,seed=1):
    rng=np.random.default_rng(seed); i=np.arange(1,N+1); Om=W0+D*np.tan(np.pi*(i/(N+1)-0.5))
    th=rng.uniform(0,2*np.pi,N); X=1e-3; V=0.0; As=[];Rs=[]
    for k in range(int(T/dt)):
        A=np.hypot(X,V/W0); Psi=np.arctan2(X,V/W0)
        acc=(G*np.sin(th).sum()-2*M*zeta*W0*V-K*X)/M; V+=dt*acc; X+=dt*V
        th+=dt*(Om+C*A*np.sin(Psi-th+alpha))
        if k*dt>T*0.7 and k%25==0: As.append(A); Rs.append(abs(np.exp(1j*th).mean()))
    return N,float(np.mean(As)),float(np.mean(Rs))
N=int(sys.argv[1]); print(json.dumps(run(N)))
