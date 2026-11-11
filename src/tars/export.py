import json, math, numpy as np, mujoco
from tars import *
from walk import run
from wheel_check import make
out={}
st,fr,d=run(15,0.02,T=20,record=True)
dt=np.diff(st[:,0])[-6:].mean(); dx=np.diff(st[:,1])[-6:].mean(); print('walk speed',dx/dt,'step',dt,'stride',dx)
out['walk']={'slope':0,'frames':np.round(fr,4).tolist(),'speed':dx/dt}
def roll(angles,slope_deg,T=15,w0=1.2):
    sl=math.radians(slope_deg); m,d=make(angles,slope=sl); l=LS/2-RF
    d.qpos[1]=l+RF-1.0; d.qvel[2]=w0; d.qvel[0]=w0*l; frames=[]; tn=0
    while d.time<T:
        mujoco.mj_step(m,d)
        if d.time>=tn: frames.append([d.time]+list(d.qpos)); tn+=1/60
    fr=np.array(frames); return fr
for name,ang in (('star',(0,45,90,135)),('cross',(0,90,0,90))):
    for sd in (6,):
        fr=roll(ang,sd); v=(fr[-1,1]-fr[-120,1])/2.0
        print(name,sd,'final speed along slope %.2f'%v,'distance %.1f'%fr[-1,1])
        out[name]={'slope':sd,'frames':np.round(fr,4).tolist(),'angles':ang}
json.dump(out,open('/home/claude/post7/tars_frames.json','w'))
import os; print(os.path.getsize('/home/claude/post7/tars_frames.json')/1e6,'MB')
