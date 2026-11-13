"""Figures for 'Twelve years of compute, seen through Gargantua'."""
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR,PU='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c','#7b4fa3'
out='../../electronics/compute-since-interstellar/'

# ---- hardware (sources in the post): TechPowerUp, hmc-tech, PassMark
GPU=[  # name, year, fp64, fp32, tensor fp16 (fp32 acc, dense), tensor fp8 dense, tensor fp4 dense, watts, price, transistors (bn)
 ('GTX Titan',2013.13,1.57,4.709,None,None,None,250,999,7.08),
 ('GTX 1080 Ti',2017.17,0.354,11.34,None,None,None,250,699,11.8),
 ('RTX 3090',2020.67,0.556,35.58,71.16,None,None,350,1499,28.3),
 ('RTX 4090',2022.78,1.29,82.58,165.2,660.6,None,450,1599,76.3),
 ('RTX 5090',2025.08,1.64,104.8,209.5,838,1680,575,1999,92.2)]
CPU=[('Xeon E5-2680 v2',2013.6,1776,12541,115),('Core i7-4790',2014.4,2226,7257,84),('Core i7-8700K',2017.8,2712,13553,95),('Ryzen 9 3900X',2019.6,2702,32463,105),
     ('Core i9-14900KS',2024.1,4805,59806,150),('Ryzen 9 9950X',2024.6,4726,65689,170),('Core Ultra 9 285K',2024.8,5087,67230,125),('Apple M4 (10-core)',2024.4,4490,23557,22)]
fig,ax=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.35,1]})
yr=[g[1] for g in GPU]
for idx,lab,c,m in [(4,'tensor FP4 (AI)',PU,'v'),(5,'tensor FP8 (AI)',BR,'D'),(4-0,'',None,None)][:2]+[(4,'',None,None)][:0]:
    pass
series=[(6,'tensor FP4 (AI)',PU,'v'),(5,'tensor FP8 (AI)',BR,'D'),(4,'tensor FP16 (AI)',C3,'s'),(3,'FP32 (graphics, rendering)',C1,'o'),(2,'FP64 (scientific)',C2,'^')]
for i,lab,c,m in series:
    xs=[g[1] for g in GPU if g[i]]; ys=[g[i] for g in GPU if g[i]]
    ax[0].semilogy(xs,ys,marker=m,color=c,lw=1.6,label=lab)
for g in GPU: ax[0].annotate(g[0],(g[1],g[3]),xytext=(0,-14),textcoords='offset points',fontsize=7.5,ha='center',color='#555')
ax[0].set_ylabel('peak throughput (trillion operations per second)'); ax[0].set_ylim(.2,4000); ax[0].legend(fontsize=8,loc='upper left')
ax[0].set_title('Nvidia\'s flagship consumer GPUs, 2013–2025',fontsize=9.5)
c0=CPU[0]
ax[1].plot([c[1] for c in CPU],[c[2]/c0[2] for c in CPU],'o',color=C1,label='single-thread speed'); ax[1].plot([c[1] for c in CPU],[c[3]/c0[3] for c in CPU],'s',color=C2,label='all cores together')
ax[1].plot([GPU[0][1],GPU[-1][1]],[1,GPU[-1][3]/GPU[0][3]],':',color=GR,label='GPU FP32, for comparison')
for c in CPU:
    if c[0] in ('Xeon E5-2680 v2','Ryzen 9 9950X','Apple M4 (10-core)','Core i7-8700K'): ax[1].annotate(c[0],(c[1],c[2]/c0[2]),xytext=(4,-12 if c[0]!='Apple M4 (10-core)' else 6),textcoords='offset points',fontsize=7.5,color='#555')
ax[1].set_yscale('log'); ax[1].set_ylabel('relative to the Xeon E5-2680 v2 that rendered Gargantua'); ax[1].legend(fontsize=8,loc='upper left'); ax[1].set_ylim(.5,40)
ax[1].set_title('CPUs: PassMark scores relative to 2013',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_hardware.png'); plt.close(fig)

# ---- precision renders
Z=np.load('precision_images.npz'); NY,NX=Z['float64_fate'].shape
def colour(k):
    f=Z[k+'_fate']; th=Z[k+'_th']; ph=Z[k+'_ph']; rd=Z[k+'_rdisk']; img=np.zeros((NY,NX,3))
    # sky: a latitude-longitude grid, so distortions are visible
    u=(np.mod(ph,2*np.pi)/(2*np.pi)*24); v=(np.clip(th,0,np.pi)/np.pi*12)
    chk=((np.floor(u)+np.floor(v))%2==0); sky=np.where(chk[...,None],[.16,.20,.30],[.05,.07,.12])
    edge=(np.abs(u-np.round(u))<.03)|(np.abs(v-np.round(v))<.03); sky=np.where(edge[...,None],[.55,.62,.8],sky)
    img[f==0]=sky[f==0]
    t=np.clip((rd-2.6)/(12-2.6),0,1); dc=np.stack([1-0.15*t,0.85-0.55*t,0.45-0.4*t],-1)*(0.55+0.45*(np.floor(rd*2)%2))[...,None]
    img[f==2]=np.nan_to_num(dc[f==2]); img[f==3]=[1,0,1]
    return img
fig,ax=plt.subplots(2,2,figsize=(11,5.8))
for a_,k,t in zip(ax.flat,['float64','float32','bfloat16','float16'],['64-bit (double precision): the reference','32-bit (single precision)','bfloat16 (an AI format, 8-bit significand)','16-bit (half precision)']):
    a_.imshow(colour(k)); a_.set_title(t,fontsize=9.5); a_.axis('off')
R=json.load(open('precision_results.json'))
ax[0,1].text(5,NY-8,'no pixel differs' if R['float32']['fate_mismatch_pct']==0 else f"{R['float32']['fate_mismatch_pct']:.3f}% of pixels differ",color='w',fontsize=8)
ax[1,0].text(5,NY-8,f"{R['bfloat16']['fate_mismatch_pct']:.1f}% of pixels differ",color='w',fontsize=8)
ax[1,1].text(NX/2,NY/2,'overflow: every value of r² is\nout of range, so every ray fails',color='w',fontsize=9,ha='center',va='center')
fig.tight_layout(); fig.savefig(out+'fig_precision.png'); plt.close(fig)

# ---- photon sphere
P=json.load(open('photonsphere.json')); e=np.array(P['eps']); orb=np.array(P['orbits_float64'])
fig,ax=plt.subplots(figsize=(8.4,4.4))
d=np.array(P['float32']['dphi']); ax.loglog(e,np.maximum(d,1e-10),'o',ms=4,color=C1,label='32-bit')
ax.loglog(e,30*2**-24/e,'--',color=C1,lw=1,alpha=.7,label='30 × 2⁻²⁴ ÷ (b/b_c − 1)')
d=np.array(P['bfloat16']['dphi']); ax.loglog(e,np.maximum(d,1e-10),'s',ms=4,color=C2,label='bfloat16')
ax.axhline(np.radians(60)/1920,color=GR,ls=':'); ax.text(1.5,np.radians(60)/1920*1.5,'one pixel, 1920 pixels across a 60° view',fontsize=8,color='#555')
for x_ in (1e-1,1e-3,1e-5,1e-7):
    o_=np.interp(np.log10(x_),np.log10(e),orb); ax.annotate(f'{o_:.1f} orbits',(x_,3e-9),fontsize=8,ha='center',color='#555')
ax.set_xlabel('how close the ray passes to the photon sphere:  b / b_c − 1'); ax.set_ylabel('error in total deflection (radians)')
ax.set_ylim(1e-9,60); ax.legend(fontsize=8.5,loc='upper left'); ax.invert_xaxis()
ax.set_title('Rounding error grows as a ray winds round the photon sphere (non-spinning hole)',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_photonsphere.png'); plt.close(fig)

# ---- thumbnail

print('ok')
