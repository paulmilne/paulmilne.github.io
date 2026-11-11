import numpy as np
from endurance import *
def rot(axis,a):
    axis=axis/np.linalg.norm(axis); K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def tumble(rpm=68,wobble_deg=5.0,T=3.0,dt=2e-4,lost=LOST):
    """Torque-free rigid body in its principal frame. Returns t, rotation matrices, body omega."""
    M,com,I,_=mass_props(lost); ev,V=np.linalg.eigh(I)            # V: ring-frame -> principal columns
    w0=rpm2w(rpm); th=np.radians(wobble_deg)
    wr=w0*np.array([np.sin(th),0,np.cos(th)])                         # ring frame, tilted toward +x
    w=V.T@wr; Rm=np.eye(3)                                           # body (principal) frame
    out_t=[];out_R=[];out_w=[]
    f=lambda w: np.array([(ev[1]-ev[2])*w[1]*w[2]/ev[0],(ev[2]-ev[0])*w[2]*w[0]/ev[1],(ev[0]-ev[1])*w[0]*w[1]/ev[2]])
    for k in range(int(T/dt)):
        if k%10==0: out_t.append(k*dt); out_R.append(Rm.copy()); out_w.append(w.copy())
        k1=f(w);k2=f(w+dt/2*k1);k3=f(w+dt/2*k2);k4=f(w+dt*k3); wn=w+dt/6*(k1+2*k2+2*k3+k4)
        wm=0.5*(w+wn); a=np.linalg.norm(wm)*dt; Rm=Rm@rot(wm,a) if a>0 else Rm; w=wn
    return np.array(out_t),np.array(out_R),np.array(out_w),(M,com,I,ev,V)
if __name__=='__main__':
    for wob in (0,5,15):
        t,Rs,ws,(M,com,I,ev,V)=tumble(68,wob)
        Lb=ev*ws[0]; Ls=Rs[0]@Lb; Lhat=Ls/np.linalg.norm(Ls)
        hub_b=V.T@(-com); n_b=V.T@np.array([0,0,1.0])
        P=np.einsum('kij,j->ki',Rs,hub_b); N=np.einsum('kij,j->ki',Rs,n_b)
        cone=np.degrees(np.arccos(np.clip(N@Lhat,-1,1)))
        dtt=t[1]-t[0]; acc=np.gradient(np.gradient(P,dtt,axis=0),dtt,axis=0)
        wsp=np.einsum('kij,kj->ki',Rs,ws); wdot=np.gradient(wsp,dtt,axis=0)
        print(f'wobble {wob}: hub axis cone {cone.min():.1f}-{cone.max():.1f} deg, port accel max {np.linalg.norm(acc,axis=1)[5:-5].max()/g0:.1f} g, |omega dot| max {np.linalg.norm(wdot,axis=1)[5:-5].max():.1f} rad/s2, along-L port motion {np.ptp(P@Lhat):.2f} m')
