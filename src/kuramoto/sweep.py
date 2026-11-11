import numpy as np, json, sys
g=0.5
def bim(N,w0):
    i=np.arange(1,N+1); u=i/(N+1)
    # half from each Lorentzian, interleaved quantiles
    h=N//2; uu=(np.arange(1,h+1))/(h+1); L=g*np.tan(np.pi*(uu-0.5))
    return np.concatenate([L+w0,L-w0])
def full_sweep(w0,Ks,N=2000,hold=80,dt=0.02,seed=0):
    w=bim(N,w0); rng=np.random.default_rng(seed); th=rng.uniform(0,2*np.pi,N); out=[]
    for K in Ks:
        rs=[]
        for k in range(int(hold/dt)):
            z=np.exp(1j*th).mean(); th+=dt*(w+K*abs(z)*np.sin(np.angle(z)-th))
            if k*dt>hold/2: rs.append(abs(z))
        rs=np.array(rs); out.append((K,rs.mean(),rs.min(),rs.max()))
    return out
def oa_sweep(w0,Ks,hold=200,dt=0.01):
    z1=0.01+0.005j; z2=0.012-0.004j; out=[]
    for K in Ks:
        rs=[]
        for k in range(int(hold/dt)):
            z=(z1+z2)/2
            d1=(1j*w0-g)*z1+K/2*(z-np.conj(z)*z1**2)
            d2=(-1j*w0-g)*z2+K/2*(z-np.conj(z)*z2**2)
            z1+=dt*d1; z2+=dt*d2
            if abs(z1)<1e-6: z1+=1e-6
            if abs(z2)<1e-6: z2+=1e-6j
            if k*dt>hold/2: rs.append(abs((z1+z2)/2))
        rs=np.array(rs); out.append((K,rs.mean(),rs.min(),rs.max()))
    return out
w0=float(sys.argv[1]); Kmax=float(sys.argv[2])
up=np.round(np.arange(0,Kmax+1e-9,Kmax/60),4); dn=up[::-1]
res={'w0':w0,'full_up':full_sweep(w0,up),'full_dn':full_sweep(w0,dn,seed=1),'oa_up':oa_sweep(w0,up),'oa_dn':oa_sweep(w0,dn)}
json.dump(res,open(f'/home/claude/post2/sweep_{w0}.json','w'))
print('done',w0)
