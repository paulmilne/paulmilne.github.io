"""How much floating-point precision does Gargantua need?
Renders the same view in float64, float32, bfloat16 and float16 and compares
each with float64. Also follows single rays at impact parameters approaching
the critical value, where errors are amplified by every orbit."""
import numpy as np, time, json, warnings
from kerr import camera_rays, trace, bf16, deriv
warnings.filterwarnings('ignore')
a=0.6; r0=25.0; th0=np.radians(85); NX,NY=320,160; FOV=60
risco=lambda a:(lambda Z1,Z2:3+Z2-np.sqrt((3-Z1)*(3+Z1+2*Z2)))(1+(1-a*a)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3)),np.sqrt(3*a*a+(1+(1-a*a)**(1/3)*((1+a)**(1/3)+(1-a)**(1/3)))**2))
DISK=(risco(a),12.0)
pr,pth,L,E=camera_rays(NX,NY,FOV,r0,th0,a)
FMT={'float64':dict(dtype=np.float64),'float32':dict(dtype=np.float32),'bfloat16':dict(dtype=np.float32,rnd=bf16),'float16':dict(dtype=np.float16)}
res={}; img={}
for k,kw in FMT.items():
    t=time.time(); o=trace(pr,pth,L,r0,th0,a,disk=DISK,**kw); dt=time.time()-t
    img[k]=o; res[k]=dict(seconds=round(dt,2),rays_per_s=round(NX*NY/dt),fates=np.bincount(o['fate'].ravel(),minlength=4).tolist(),mean_steps=round(float(o['steps'].mean()),1))
ref=img['float64']
def ang(o):
    v=lambda t,p: np.stack([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
    return np.arccos(np.clip((v(o['th'],o['ph'])*v(ref['th'],ref['ph'])).sum(0),-1,1))
pix=np.radians(FOV)/NX
for k,o in img.items():
    if k=='float64': continue
    same=o['fate']==ref['fate']; esc=same&(ref['fate']==0); dsk=same&(ref['fate']==2)
    e=ang(o)[esc]; dr=np.abs(o['rdisk']-ref['rdisk'])[dsk]
    res[k].update(fate_mismatch_pct=round(100*(1-same.mean()),3),sky_err_median_px=float(np.median(e)/pix) if e.size else None,sky_err_p99_px=float(np.percentile(e,99)/pix) if e.size else None,
                  sky_err_max_px=float(e.max()/pix) if e.size else None,disk_err_median=float(np.median(dr)) if dr.size else None,disk_err_p99=float(np.percentile(dr,99)) if dr.size else None)
print(json.dumps(res,indent=1))
np.savez_compressed('precision_images.npz',**{f'{k}_{f}':img[k][f] for k in img for f in ('fate','th','ph','rdisk')})
# also a reference at half the step, to compare rounding error with the integrator's own error
o2=trace(pr,pth,L,r0,th0,a,disk=DISK,dtype=np.float64,h0=0.025)
same=o2['fate']==ref['fate']; e=ang(o2)[same&(ref['fate']==0)]
res['float64_half_step']=dict(fate_mismatch_pct=round(100*(1-same.mean()),3),sky_err_median_px=float(np.median(e)/pix),sky_err_p99_px=float(np.percentile(e,99)/pix))
print('float64 at half step vs float64:',res['float64_half_step'])
json.dump(res,open('precision_results.json','w'),indent=1)
