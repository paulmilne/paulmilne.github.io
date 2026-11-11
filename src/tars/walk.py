import math, numpy as np, mujoco
from tars import *
A=(0,3); B=(1,2)
def run(theta_deg=15, push=0.04, clear=0.06, g=G, T=12.0, v0=0.6, record=False, rec_dt=1/60):
    th=math.radians(theta_deg)
    m=mujoco.MjModel.from_xml_string(xml(g=g)); d=mujoco.MjData(m)
    l=LS/2-RF
    # start: stance A at psi=-th*0.5 (foot ahead), swing B mirrored and lifted
    psi0=-0.5*th
    d.qpos[2]=0.0
    for i in A: d.qpos[3+2*i]=psi0
    for i in B: d.qpos[3+2*i]=-psi0; d.qpos[4+2*i]=clear
    d.qpos[1]=l*math.cos(psi0)+RF-1.0
    d.qvel[0]=v0; 
    for i in A: d.qvel[3+2*i]=v0/l*math.cos(psi0)/1.0
    mujoco.mj_forward(m,d)
    stance,swing=A,B; steps=[]; frames=[]; last_step_t=0; tnext=0
    floor=0
    def foot_on_ground(idx):
        ids={m.geom(f'footA{i}').id for i in idx}
        return any((c.geom1 in ids or c.geom2 in ids) for c in d.contact[:d.ncon])
    while d.time<T:
        pitch=d.qpos[2]
        psi_s=pitch+np.mean([d.qpos[3+2*i] for i in stance])
        s=min(1,max(0,(psi_s+th)/(2*th)))
        # swing mirrors stance; relative hinge targets (axle-relative) so that absolute swing = -psi_s
        for i in stance: d.ctrl[2*i]=d.qpos[3+2*i]          # stance hinge: hold current (no torque about axle)
        for i in swing:  d.ctrl[2*i]=-psi_s-pitch
        lift=clear if s<0.85 else clear*max(0,(1-s)/0.15)
        for i in swing: d.ctrl[2*i+1]=lift
        pz=-push*max(0,(s-0.6)/0.4)                           # push-off: stance slab extends downward late in stance
        for i in stance: d.ctrl[2*i+1]=pz
        mujoco.mj_step(m,d)
        if s>0.8 and foot_on_ground(swing) and d.time-last_step_t>0.15:
            steps.append((d.time,d.qpos[0])); last_step_t=d.time; stance,swing=swing,stance
        if d.qpos[1]+1.0<0.35: break   # fell
        if record and d.time>=tnext:
            frames.append([d.time]+list(d.qpos)); tnext+=rec_dt
    return np.array(steps),np.array(frames),d
if __name__=='__main__':
    for th in (10,15,20):
        for push in (0.0,0.02,0.04,0.06):
            st,_,d=run(th,push)
            if len(st)>4:
                dt=np.diff(st[:,0])[-4:].mean(); dx=np.diff(st[:,1])[-4:].mean()
                print(th,push,'steps',len(st),'t',round(d.time,1),'step time %.2f stride %.2f speed %.2f'%(dt,dx,dx/dt))
            else: print(th,push,'steps',len(st),'t',round(d.time,1),'hub z',round(d.qpos[1]+1,2))
