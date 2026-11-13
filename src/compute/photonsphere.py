"""Single equatorial rays past a non-spinning hole, at impact parameters approaching the
critical value b_c = 3*sqrt(3). Compares the total deflection computed in float32 and
bfloat16 with float64 at the same step size, so only rounding differs."""
import numpy as np, json
from kerr import trace, bf16
a=0.0; r0=50.0; bc=3*np.sqrt(3)
eps=np.logspace(-7,0,57); b=bc*(1+eps)
L=b.copy(); D=r0**2-2*r0; P=r0**2-0*L
pr=-np.sqrt(np.maximum(P**2/D-L**2,0))/np.sqrt(D)        # H = 0 at theta = pi/2, inbound
pth=np.zeros_like(b); th0=np.pi/2
out={}
for k,kw in {'float64':dict(dtype=np.float64),'float32':dict(dtype=np.float32),'bfloat16':dict(dtype=np.float32,rnd=bf16)}.items():
    o=trace(pr[None],pth[None],L[None],r0,th0,a,r_far=r0+0.5,max_steps=20000,h0=0.02,**kw); out[k]=o
ph64=out['float64']['ph'][0]; orbits=(ph64-np.pi)/(2*np.pi)
R={'eps':eps.tolist(),'orbits_float64':orbits.tolist(),'fate64':out['float64']['fate'][0].tolist()}
for k in ('float32','bfloat16'):
    R[k]=dict(dphi=np.abs(out[k]['ph'][0]-ph64).tolist(),fate=out[k]['fate'][0].tolist())
json.dump(R,open('photonsphere.json','w'))
for e,o,d32,d16,f in zip(eps[::4],orbits[::4],R['float32']['dphi'][::4],R['bfloat16']['dphi'][::4],out['bfloat16']['fate'][0][::4]): print(f'b/bc-1={e:.1e}  orbits {o:6.2f}  |dphi| f32 {d32:.1e}  bf16 {d16:.1e} (fate {f})')
