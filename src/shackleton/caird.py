"""Monte Carlo of the James Caird voyage: dead reckoning, sparse sun sights, and aiming off."""
import numpy as np
NM=1852.0
LAT0=-58.0; COS0=np.cos(np.radians(LAT0))
def xy(lat,lon): return np.array([(lon+54.87)*60*COS0,(lat+61.10)*60])   # nmi east/north of Point Wild
PW=xy(-61.10,-54.87)
# simplified South Georgia outline (lat, lon), clockwise from the NW tip
SG_LL=[(-54.00,-38.05),(-54.02,-37.70),(-54.08,-37.30),(-54.10,-37.05),(-54.16,-36.71),(-54.25,-36.45),(-54.40,-36.20),(-54.55,-36.00),(-54.80,-35.78),
       (-54.88,-36.05),(-54.75,-36.35),(-54.60,-36.65),(-54.45,-36.95),(-54.30,-37.15),(-54.22,-37.32),(-54.15,-37.60),(-54.08,-37.90)]
SG=np.array([xy(a,b) for a,b in SG_LL])
NWTIP=SG[0]; SETIP=SG[8]
KHB=xy(-54.20,-37.30)          # King Haakon Bay (south-west coast)
STROM=xy(-54.16,-36.71)        # Stromness (north coast)
# the south-west coast, from NW tip to SE tip, for aim points
SWC=np.vstack([SG[0]]+[SG[i] for i in range(len(SG)-1,7,-1)])
def coast_point(d):
    seg=np.diff(SWC,axis=0); L=np.hypot(*seg.T); c=np.concatenate([[0],np.cumsum(L)])
    i=min(np.searchsorted(c,d)-1,len(L)-1); i=max(i,0); f=(d-c[i])/L[i]; return SWC[i]+f*seg[i]
COAST_LEN=np.hypot(*np.diff(SWC,axis=0).T).sum()
from matplotlib.path import Path
SGPATH=Path(SG)
SIGHT_DAYS=(2,9,10,13)          # 26 Apr, 3, 4, 7 May (days after noon 24 Apr)
def voyage(n=2000,aim=30.0,sight_sig=5.0,sights=SIGHT_DAYS,hdg_sig=6.0,spd_sig=0.12,cur=(8.0,4.0),days=22,seed=0,keep=0):
    """aim: nmi down the south-west coast from the NW tip. Returns outcomes and optional tracks."""
    rng=np.random.default_rng(seed)
    target=coast_point(aim)
    dt=0.25; steps=int(days/dt)
    true=np.tile(PW,(n,1)).astype(float); est=true.copy()
    bias=rng.normal(0,hdg_sig,n)                      # steady compass/leeway bias (deg)
    curv=np.column_stack([rng.normal(cur[0],cur[1],n),rng.normal(0,cur[1],n)])   # unknown current, nmi/day
    speed=48.0                                        # nmi/day through the water
    out=np.zeros(n,int)                               # 0 at sea, 1 landfall, 2 passed north, 3 passed south
    tracks=[]; done_day=np.full(n,np.nan)
    for k in range(steps):
        t=k*dt
        # course: north for two days, then steer for the target from the estimated position
        if t<2: h=np.zeros(n)
        else:
            d=target-est; h=np.clip(np.degrees(np.arctan2(d[:,0],d[:,1])),0,150)   # cannot beat back west into the westerlies
        he=h+bias+rng.normal(0,hdg_sig,n)*np.sqrt(dt)
        sp=speed*(1+rng.normal(0,spd_sig,n))
        vt=np.column_stack([sp*np.sin(np.radians(he)),sp*np.cos(np.radians(he))])+curv
        ve=np.column_stack([speed*np.sin(np.radians(h)),speed*np.cos(np.radians(h))])
        live=out==0
        true[live]+=vt[live]*dt; est[live]+=ve[live]*dt
        for sd in sights:
            if abs(t-sd)<dt/2: est[live]=true[live]+rng.normal(0,sight_sig,(live.sum(),2))
        inside=SGPATH.contains_points(true)
        hit=live&inside; out[hit]=1; done_day[hit]=t
        # passed the island: east of its far end
        axis_y=NWTIP[1]+(true[:,0]-NWTIP[0])*(SETIP[1]-NWTIP[1])/(SETIP[0]-NWTIP[0])
        past=live&~inside&((true[:,0]>SETIP[0]+10)|((true[:,0]>NWTIP[0]+10)&(true[:,1]>axis_y+25)))
        out[past&(true[:,1]>axis_y)]=2; out[past&(true[:,1]<=axis_y)]=3
        if keep: tracks.append(true[:keep].copy())
        if not (out==0).any(): break
    return out,np.array(tracks) if keep else None,done_day,true
if __name__=='__main__':
    print('coast length nmi',COAST_LEN,'PW->KHB',np.hypot(*(KHB-PW)),'PW->NW tip',np.hypot(*(NWTIP-PW)))
    for aim in (0,15,30,45,60,75):
        o,_,dd,_=voyage(aim=aim)
        print(f'aim {aim:3d}: landfall {np.mean(o==1):.2f} north {np.mean(o==2):.2f} south {np.mean(o==3):.2f} at sea {np.mean(o==0):.2f} median day {np.nanmedian(dd):.1f}')
    for s in ((),(2,),(2,9,10),SIGHT_DAYS):
        o,_,_,_=voyage(aim=30,sights=s); print('sights',s,'landfall',np.mean(o==1))
