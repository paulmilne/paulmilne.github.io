import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from endurance import *; from tumble import tumble
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
COL={'engine':'#d9622b','cargo':'#b08d3c','habitat':'#1f5fa8','cryo':'#6a4c93','command':'#2a9d6f'}
out='/home/claude/blog/sci-fi/endurance-docking/'
# F1 ring before/after
fig,ax=plt.subplots(1,2,figsize=(10,4.9))
for a,lost,title in ((ax[0],(),'intact: centre of mass at the hub'),(ax[1],LOST,'after the explosion')):
    M,com,I,mods=mass_props(lost); a.set_aspect('equal'); a.axis('off')
    for k in range(4):
        ang=np.radians(45+90*k); a.plot([0,(R-5)*np.cos(ang)],[0,(R-5)*np.sin(ang)],color='#cccccc',lw=3,zorder=0)
    a.add_patch(Circle((0,0),R,fill=False,color='#dddddd',lw=6,zorder=0))
    for k,t in enumerate(LAYOUT):
        ang=np.radians(30*k); x,y=R*np.cos(ang),R*np.sin(ang)
        tr=Affine2D().rotate_around(x,y,ang)+a.transData
        if k in lost:
            a.add_patch(Rectangle((x-4,y-5),8,10,fill=False,ec='#bbbbbb',ls='--',transform=tr))
        else:
            a.add_patch(Rectangle((x-4,y-5),8,10,color=COL[t],transform=tr))
    a.add_patch(Circle((0,0),3,color='#555555'))
    a.plot(*com[:2],'x',color='k',ms=10,mew=2.5,zorder=5)
    if lost:
        d=np.linalg.norm(com); a.add_patch(Circle(com[:2],d,fill=False,color='k',ls=':',lw=1))
        a.annotate('new centre of mass\n= new spin axis',xy=com[:2],xytext=(-27,-38),fontsize=8.5,arrowprops=dict(arrowstyle='->'))
        a.annotate('docking port (hub)\ncircles it at 5.9 m',xy=(0,0),xytext=(8,-40),fontsize=8.5,arrowprops=dict(arrowstyle='->'))
        a.text(R+4,6,'lost: engine\nand cargo',fontsize=8.5,color=GR)
    a.set_xlim(-42,42); a.set_ylim(-44,40); a.set_title(title,fontsize=10)
for t,c in COL.items(): ax[0].plot([],[],'s',color=c,label=t)
ax[0].legend(loc='lower left',fontsize=8,ncol=3,bbox_to_anchor=(0,-0.08))
fig.tight_layout(); fig.savefig(out+'fig_ring.png'); plt.close(fig)
# F2 g against distance from the axis
r=np.geomspace(0.1,40,200)
fig,a=plt.subplots(figsize=(7,4))
for rpm,c in ((5.5,C3),(20,C1),(68,C2)):
    w=rpm2w(rpm); a.loglog(r,w*w*r/g0,color=c,lw=2,label=f'{rpm} rpm')
a.axvspan(0.5,1.5,color=BR,alpha=.15); a.text(0.55,300,"Cooper's seat\n(port overhead)",fontsize=8,color=BR)
a.axvline(R,color=GR,ls='--',lw=1); a.text(R*0.6,300,'ring\nmodules',fontsize=8,color=GR,ha='right')
a.axhline(1,color='k',lw=.6); a.text(0.11,1.15,'1 g',fontsize=8)
a.set_xlabel('distance from the spin axis (m)'); a.set_ylabel('centripetal acceleration (g)'); a.legend(loc='lower right',fontsize=8.5); a.set_ylim(0.01,1000)
fig.tight_layout(); fig.savefig(out+'fig_gload.png'); plt.close(fig)
# F3 wobble: approach point 10 m out along the docking axis, seen along the spin axis
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for wob,c in ((0,C1),(5,C2)):
    t,Rs,ws,(M,com,I,ev,V)=tumble(68,wob,T=2.0)
    Lb=ev*ws[0]; Ls=Rs[0]@Lb; z=Ls/np.linalg.norm(Ls); x=np.cross(z,[0,1,0]); x/=np.linalg.norm(x); y=np.cross(z,x)
    P=np.einsum('kij,j->ki',Rs,V.T@(-com)); N=np.einsum('kij,j->ki',Rs,V.T@np.array([0,0,1.]))
    Q=P+10*N
    ax[0].plot(Q@x,Q@y,color=c,lw=1,label=f'spin tilted {wob}° off the hub axis')
    ax[1].plot(t,np.degrees(np.arccos(np.clip(N@z,-1,1))),color=c,lw=1.2)
ax[0].plot(0,0,'x',color='k',ms=9,mew=2); ax[0].set_aspect('equal'); ax[0].set_xlabel('m'); ax[0].set_ylabel('m')
ax[0].set_title('a point 10 m out along the docking axis,\nseen along the spin axis, 68 rpm, first 2 s',fontsize=9.5); ax[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.0))
ax[0].set_xlim(-8,8); ax[0].set_ylim(-8,10)
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('docking axis off the spin axis (°)'); ax[1].set_title('wobble of the docking axis',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_port.png'); plt.close(fig)
# F4 required Ranger acceleration vs rpm
M,com,I,_=mass_props(LOST); d=np.linalg.norm(com)
rpm=np.linspace(0,75,300); w=rpm2w(rpm)
fig,a=plt.subplots(figsize=(7,4))
a.semilogy(rpm,np.maximum(w*w*d/g0,1e-3),color='k',lw=2,label='to hold station on the port (5.9 m off the spin axis)')
for acc,c,lab in ((2.5,C2,'Ranger main engines, assumed 2.5 g'),(0.1,C1,'manoeuvring thrusters, assumed 0.1 g')):
    a.axhline(acc,color=c,ls='--',lw=1.2,label=lab); wr=np.sqrt(acc*g0/d)*60/2/np.pi; a.plot([wr],[acc],'o',color=c); a.text(wr+1,acc*0.62,f'{wr:.0f} rpm',color=c,fontsize=8.5)
for r,lab in ((68,'dialogue: 68 rpm'),(20,'on screen: ~20 rpm'),(5.5,'design: 5.5 rpm')):
    a.axvline(r,color=GR,lw=.8,ls=':'); a.text(r+0.7,0.0016,lab,rotation=90,fontsize=8,color=GR)
a.set_xlabel('Endurance spin (rpm)'); a.set_ylabel('acceleration the Ranger needs (g)'); a.set_ylim(1e-3,100); a.set_xlim(0,75)
a.legend(fontsize=8,loc='upper left')
fig.tight_layout(); fig.savefig(out+'fig_station.png'); plt.close(fig)
print('d',d)
