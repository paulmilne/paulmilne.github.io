import math, numpy as np, mujoco
from tars import *
M=4*MS+4+0.4; l=LS/2-RF
def make(angles,slope=0.0,g=G,extra=0.0):
    m=mujoco.MjModel.from_xml_string(xml(angles,g=g,slope=slope,extra_hub=extra)); d=mujoco.MjData(m)
    for i,a in enumerate(angles): d.qpos[3+2*i]=math.radians(a); d.ctrl[2*i]=math.radians(a); d.ctrl[2*i+1]=0
    return m,d
def footfall(angles,w=2.0,extra=0.0):
    n=8 if len(set(angles))==4 else 4; a=math.pi/n
    m,d=make(angles,extra=extra)
    th=a-0.02   # rotate whole robot: pitch; spoke 'footA0' points down at pitch 0
    d.qpos[2]=th; d.qpos[1]=(l+RF)*math.cos(th)+0.0 - 1.0 + 0.0  # z slide relative to body pos 1.0: hub height = l cos th + RF
    d.qpos[1]=l*math.cos(th)+RF-1.0
    d.qvel[0]=w*l*math.cos(th); d.qvel[1]=-w*l*math.sin(th); d.qvel[2]=w
    mujoco.mj_forward(m,d)
    w0=None; t_hit=None
    while d.time<0.5:
        wb=d.qvel[2]; mujoco.mj_step(m,d)
        feet=[c for c in d.contact[:d.ncon]]
        xs=[d.geom_xpos[c.geom2 if c.geom1==0 else c.geom1][0] for c in feet]
        if t_hit is None and any(x>d.qpos[0]+0.05 for x in xs): t_hit=d.time; w0=wb
        elif t_hit and d.time>t_hit+0.01: return d.qvel[2]/w0
    return float('nan')
def roll(angles,slope,T=12.0,w0=1.0,extra=0.0):
    m,d=make(angles,slope=slope,extra=extra)
    d.qpos[1]=l+RF-1.0; d.qvel[2]=w0; d.qvel[0]=w0*l
    out=[]
    while d.time<T:
        mujoco.mj_step(m,d)
        if int(d.time/2e-4)%50==0: out.append((d.time,d.qvel[2],d.qvel[0]))
    return np.array(out)
if __name__=='__main__':
    I=4*MS*(LS**2+D**2)/12; k2=I/(M*(LS/2)**2)
    for ang,n in (((0,45,90,135),8),((0,90,0,90),4)):
        print(n,'spokes: footfall ratio MuJoCo',round(footfall(ang),3),'theory',round(ratio(n,k2),3))
    o=roll((0,45,90,135),math.radians(4))
    print('star on 4 deg slope: mean w last 4 s',o[o[:,0]>8,1].mean(),'speed',o[o[:,0]>8,2].mean())
