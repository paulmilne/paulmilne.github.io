import numpy as np, json
M=1.13e5; f0=1.03; W0=2*np.pi*f0; K=M*W0**2; zeta=0.0075; G=25.0; C=16.0; alpha=np.pi/2; D=0.072*W0
Nc=4*zeta*K*D/(G*C); print('K',K,'Nc',Nc,'Delta Hz',D/2/np.pi)
def crowd_test(Nmax=260,group=10,every=60.0,dt=0.004,seed=5):
    rng=np.random.default_rng(seed)
    i=np.arange(1,Nmax+1); Om=W0+D*np.tan(np.pi*(i/(Nmax+1)-0.5)); Om=rng.permutation(Om)
    th=rng.uniform(0,2*np.pi,Nmax); X=0.0; V=0.0
    T=every*(Nmax//group); n=int(T/dt); rec=[]
    for k in range(n):
        t=k*dt; N=min(Nmax,group*(1+int(t//every)))
        A=np.hypot(X,V/W0); Psi=np.arctan2(X,V/W0)    # X=A sin Psi, V/W0=A cos Psi
        F=G*np.sin(th[:N]).sum()
        acc=(F-2*M*zeta*W0*V-K*X)/M
        V+=dt*acc; X+=dt*V                             # semi-implicit Euler
        th[:N]+=dt*(Om[:N]+C*A*np.sin(Psi-th[:N]+alpha))
        th[N:]+=dt*Om[N:]
        if k%25==0:
            z=np.exp(1j*th[:N]).mean(); rec.append((t,N,A,abs(z)))
    return np.array(rec)
def oa_steady(N,T=3000,dt=0.02):
    w=1e-3+0j; z=0.01+0j; gb=zeta*W0; amps=[]
    # rotating frame: remove i*W0
    for k in range(int(T/dt)):
        dw=-gb*w-1j*G*N/(2*M*W0)*z
        dz=-D*z+C/2*(np.exp(1j*alpha)*w-np.exp(-1j*alpha)*np.conj(w)*z*z)
        w+=dt*dw; z+=dt*dz
    return abs(w),abs(z)
if __name__=='__main__':
    rec=crowd_test(); np.save('crowd.npy',rec)
    Ns=np.arange(100,401,10); S=[oa_steady(N) for N in Ns]
    json.dump({'Nc':Nc,'Ns':Ns.tolist(),'A':[s[0] for s in S],'R':[s[1] for s in S]},open('oa.json','w'))
    print('final A mm',rec[-1,2]*1000,'R',rec[-1,3]); print(list(zip(Ns[::5],np.round(np.array([s[0] for s in S])[::5]*1000,1))))
