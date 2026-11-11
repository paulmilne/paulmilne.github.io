import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ipm import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/explainers/inverted-pendulum-walker/'
# --- 1. diagram, frontal plane
fig,ax=plt.subplots(1,2,figsize=(9,3.9),gridspec_kw={'width_ratios':[1,1.5]})
a=ax[0]; a.set_aspect('equal'); a.axis('off')
u=0.0; y=-0.25; h=1.0
a.plot([-0.7,0.6],[0,0],color='k',lw=1.5); a.fill_between([-0.7,0.6],[0,0],[-0.05,-0.05],color='#dddddd')
a.plot([u,y],[0,h],color=GR,lw=3); a.add_patch(plt.Circle((y,h),0.06,color=C1,zorder=3))
a.text(y-0.1,h+0.08,'centre of mass\n(height L)',ha='center',fontsize=8.5,color=C1)
a.plot([u],[0],'^',color=C2,ms=10); a.text(u+0.04,-0.15,'foot = centre of pressure u',fontsize=8.5,color=C2)
a.annotate('',xy=(y+0.35,h),xytext=(y,h),arrowprops=dict(arrowstyle='->',color=C1,lw=1.5)); a.text(y+0.12,h+0.05,'ẏ',color=C1)
xi=-0.16; a.plot([xi],[0],'o',color=BR,ms=7); a.text(xi-0.02,0.07,'ξ = y + ẏ/ω₀',fontsize=8.5,color=BR,ha='right')
a.annotate('',xy=(u,-0.06),xytext=(xi,-0.06),arrowprops=dict(arrowstyle='<->',color='k',lw=.8)); a.text((u+xi)/2,-0.25,'b_min',ha='center',fontsize=8.5)
a.text(-0.45,0.45,'gravity pulls the\nbody away from u:\nÿ = ω₀²(y − u)',fontsize=8.5,color='k')
a.set_xlim(-0.75,0.65); a.set_ylim(-0.3,1.25); a.set_title('frontal plane, right foot in stance',fontsize=9)
# rigid ground gait
o,w=walk(4.0,0.9); t=o[:,0]; off=o[:,1].mean(); o[:,1:4]-=off
b=ax[1]; b.plot(t,o[:,1]*100,color=C1,lw=1.6,label='centre of mass y'); b.step(t,o[:,2]*100,where='post',color=C2,lw=1.4,label='foot u')
b.plot(t,o[:,3]*100,color=BR,lw=1,label='extrapolated CoM ξ'); b.set_ylabel('sideways position (cm)'); b.set_xlabel('time (s)')
b2=b.twinx(); b2.plot(t,o[:,4],color=GR,lw=.8,alpha=.7); b2.set_ylabel('force on ground (N)',color=GR); b2.grid(False); b2.spines['right'].set_visible(True); b2.set_ylim(-60,60)
b.plot([],[],color=GR,lw=.8,label='sideways force on ground'); b.legend(fontsize=7.5,loc='upper center',ncol=2,bbox_to_anchor=(0.5,1.17)); b.set_ylim(-9,9); b.set_xlim(0,4)
fig.tight_layout(); fig.savefig(out+'fig_model.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
# --- 2. damping vs bridge frequency
fig,a=plt.subplots(figsize=(7,4.2))
for v,c,lab in (('abs',C1,'foot placed using absolute sway velocity'),('rel',C2,'foot placed using velocity relative to the deck')):
    R=np.array(json.load(open(f'curve_{v}.json'))); s=np.abs(R[:,0]-0.9)>0.01
    a.plot(R[s,0],R[s,1],'o-',ms=3,color=c,label=lab)
a.axhline(0,color='k',lw=.8); a.axvline(0.9,color=GR,ls=':',lw=1); a.text(0.905,140,'walker\'s own\nlateral frequency',fontsize=8,color=GR)
a.axvline(1.03,color=C3,ls='--',lw=1); a.text(1.04,-230,'Millennium Bridge\nnorth span mode',fontsize=8,color=C3)
a.fill_between([0.3,1.6],-260,0,color=C2,alpha=.05); a.text(0.32,-245,'negative damping: walker feeds energy in',fontsize=8,color=C2)
a.set_xlabel('bridge frequency (Hz)'); a.set_ylabel('damping added per walker (N s/m)'); a.set_ylim(-260,180); a.set_xlim(0.3,1.6)
a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=2)
fig.tight_layout(); fig.savefig(out+'fig_damping.png'); plt.close(fig)
# --- 3. bridge with N inverted-pendulum walkers
fig,a=plt.subplots(figsize=(7,3.8))
for N,c in ((110,GR),(140,C3),(160,C1),(200,C2)):
    o=np.load(f'bip_{N}.npy'); a.semilogy(o[:,0]/60,o[:,1]*1000,color=c,lw=1,label=f'N = {N}')
a.set_xlabel('time (minutes)'); a.set_ylabel('sway amplitude (mm)'); a.legend(fontsize=8,loc='upper left'); a.set_xlim(0,10); a.set_ylim(0.3,400)
fig.tight_layout(); fig.savefig(out+'fig_bridge.png'); plt.close(fig)
