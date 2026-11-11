import json, math, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from tars import *
from tars_numbers import roll_power, drag_power, k2, M0, l, GM
from wheel_check import footfall
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/sci-fi/tars-walking/'
# F1 configurations
fig,ax=plt.subplots(1,4,figsize=(11,3.4))
confs=[('standing',(0,0,0,0)),('crutch walk',(-15,15,15,-15)),('cross: 4 spokes',(0,90,0,90)),('star: 8 spokes',(0,45,90,135))]
for a,(name,angs) in zip(ax,confs):
    a.set_aspect('equal'); a.axis('off'); a.axhline(-0.82 if name!='crutch walk' else -0.8*math.cos(math.radians(15)),color='k',lw=1)
    for j,ang in enumerate(angs):
        tr=Affine2D().rotate_deg_around(0,0,ang)+a.transData
        a.add_patch(Rectangle((-0.11,-0.8),0.22,1.6,fc=['#c9c9c4','#a9a9a4','#a9a9a4','#c9c9c4'][j],ec='#555',lw=.8,alpha=.85,transform=tr))
    a.plot(0,0,'o',color='k',ms=4); a.set_xlim(-1,1); a.set_ylim(-1,1); a.set_title(name,fontsize=10)
ax[1].text(0,-1.0,'outer pair and inner pair\nswing alternately',ha='center',fontsize=8,color=GR)
fig.tight_layout(); fig.savefig(out+'fig_configs.png'); plt.close(fig)
# F2 energy kept per footfall
ns=np.array([3,4,5,6,8,10,12]); fig,a=plt.subplots(figsize=(6.6,3.8))
for K,c,lab in ((0.0,GR,'all mass at the hub'),(k2(M0),C1,'TARS: mass along the slabs'),(k2(M0+80),C3,'TARS carrying Brand')):
    r=np.array([ratio(n,K) for n in ns]); kept=np.where(r>0,r*r,0)
    a.plot(ns,kept,'o-',color=c,label=lab)
for n,ang in ((4,(0,90,0,90)),(8,(0,45,90,135))):
    r=footfall(ang); a.plot(n,r*r,'*',ms=13,color=C2,zorder=5)
a.plot([],[],'*',color=C2,ms=10,label='MuJoCo, one footfall')
a.set_xlabel('number of spokes'); a.set_ylabel('fraction of energy kept per footfall'); a.set_ylim(0,1); a.legend(fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(out+'fig_footfall.png'); plt.close(fig)
# F3 slope roll
F=json.load(open('tars_frames.json'))
fig,a=plt.subplots(figsize=(6.6,3.4))
for k,c,lab in (('star',C1,'star, 8 spokes'),('cross',C2,'cross, 4 spokes')):
    fr=np.array(F[k]['frames']); t=fr[:,0]; x=fr[:,1]; v=np.gradient(x,t)
    a.plot(t,np.convolve(v,np.ones(5)/5,'same'),color=c,lw=1.2,label=lab)
a.set_xlabel('time (s)'); a.set_ylabel('speed down the slope (m/s)'); a.set_title('Rolling down a 6° slope from the same push (MuJoCo)',fontsize=9.5); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(out+'fig_slope.png'); plt.close(fig)
# F4 power
v=np.linspace(0.2,12,200)
fig,a=plt.subplots(figsize=(7,4))
a.semilogy(v,[roll_power(x)[0]/1e3 for x in v],color=C1,lw=2,label='footfall losses, star')
a.semilogy(v,[roll_power(x,M0+80)[0]/1e3 for x in v],color=C3,lw=2,label='footfall losses, carrying Brand')
a.semilogy(v,drag_power(v)/1e3,color=C2,lw=2,label='water drag, 0.5 m deep')
vb=math.sqrt(GM/l)*l*math.sin(math.pi/8)/(math.pi/8); a.axvspan(vb,12,color=GR,alpha=.1); a.text(vb+0.2,0.012,'too fast to stay on the\nground between footfalls',fontsize=8,color=GR)
vw=math.sqrt(0.5*GM*l); a.axvline(vw,color='k',ls=':',lw=1); a.text(vw-0.15,300,'walking\nlimit',fontsize=8,ha='right')
a.axhline(100,color=GR,lw=.8,ls='--'); a.text(0.3,120,'a family car, ~100 kW',fontsize=8,color=GR)
a.set_xlabel('speed (m/s)'); a.set_ylabel('power needed (kW)'); a.set_ylim(0.01,1000); a.set_xlim(0,12); a.legend(fontsize=8,loc='lower right')
a.set_title("Keeping the star rolling on Miller's planet (1.3 g)",fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_power.png'); plt.close(fig)
