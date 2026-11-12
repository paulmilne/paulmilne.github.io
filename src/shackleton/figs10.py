import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from caird import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'axes.grid':True,'grid.alpha':.25,'legend.frameon':False})
C1,C2,C3,GR,BR='#1f5fa8','#d9622b','#2a9d6f','#888888','#b08d3c'
out='/home/claude/blog/astronomy/caird-navigation/'
o,tr,dd,_=voyage(n=400,aim=30,keep=120,seed=3)
fig,a=plt.subplots(figsize=(9,6))
a.fill(*SG.T,color='#c9c2ae',ec='#6b6450',zorder=3); a.text(SG[:,0].mean()+5,SG[:,1].mean()+22,'South Georgia',fontsize=9,ha='center')
a.plot(*PW,'o',color='k',zorder=4); a.text(PW[0]+8,PW[1]-12,'Point Wild,\nElephant Island',fontsize=8.5)
col={0:GR,1:C1,2:C2,3:BR}
for i in range(tr.shape[1]):
    a.plot(tr[:,i,0],tr[:,i,1],color=col[o[i]],lw=.6,alpha=.55)
tgt=coast_point(30); a.plot(*tgt,'*',color='k',ms=12,zorder=5); a.annotate('aim point, 30 nmi\ndown the coast',tgt,xytext=(tgt[0]-230,tgt[1]-10),fontsize=8.5,arrowprops=dict(arrowstyle='->'))
a.plot(*KHB,'v',color=C3,ms=7,zorder=5); a.text(KHB[0]+4,KHB[1]-26,'King Haakon Bay',fontsize=8,ha='left',color=C3)
for k,c,lab in ((1,C1,'landfall'),(2,C2,'passed north of the island'),(3,BR,'passed south')): a.plot([],[],color=c,label=lab)
a.legend(fontsize=8.5,loc='lower right'); a.set_aspect('equal'); a.set_xlabel('nautical miles east of Point Wild'); a.set_ylabel('nautical miles north')
a.set_title('120 simulated voyages: dead reckoning with four sun sights, aiming 30 nmi down the coast',fontsize=9.5)
fig.tight_layout(); fig.savefig(out+'fig_tracks.png'); fig.savefig(out+'thumbnail.png'); plt.close(fig)
aims=np.arange(0,61,5); res=np.array([[np.mean(voyage(n=3000,aim=x,seed=1)[0]==k) for k in (1,2,3)] for x in aims])
fig,a=plt.subplots(figsize=(6.8,3.8))
a.plot(aims,res[:,0],'o-',color=C1,label='landfall'); a.plot(aims,res[:,1],'o-',color=C2,label='missed: passed north'); a.plot(aims,res[:,2],'o-',color=BR,label='missed: passed south')
a.axvline(30,color='k',ls=':',lw=1); a.text(31,0.5,"Shackleton's choice",fontsize=8.5)
a.set_xlabel('aim point: nautical miles down the south-west coast from the north-west tip'); a.set_ylabel('probability'); a.legend(fontsize=8); a.set_ylim(0,1)
fig.tight_layout(); fig.savefig(out+'fig_aim.png'); plt.close(fig)
print('aim results',dict(zip(aims,res[:,0].round(2))))
sig=[2,5,10,15,20,30]; cases=[((),'no sights'),((2,),'26 Apr only'),((2,9,10),'26 Apr, 3 and 4 May'),(SIGHT_DAYS,'all four')]
fig,a=plt.subplots(figsize=(6.8,3.8))
for (sd,lab),c in zip(cases,(GR,BR,C3,C1)):
    a.plot(sig,[np.mean(voyage(n=2000,aim=30,sights=sd,sight_sig=s,seed=2)[0]==1) for s in sig],'o-',color=c,label=lab)
a.set_xlabel('error of each sun sight (nautical miles, 1σ)'); a.set_ylabel('probability of landfall'); a.set_ylim(0,1); a.legend(fontsize=8,loc='lower left')
fig.tight_layout(); fig.savefig(out+'fig_sights.png'); plt.close(fig)
