import numpy as np
g=9.81; L=1.30; Om=np.sqrt(g/L); bmin=0.0157; m=75.0
def walk(T_end, fp, X=None, Xd=None, variant='rel', dt=1e-3, phase0=0.0, rec=False):
    """Inverted-pendulum walker (Macdonald/Hof). fp: lateral gait frequency (Hz) = half step frequency.
    X(t),Xd(t): deck displacement/velocity functions (absolute). Returns force on deck history."""
    T=1/(2*fp); w=bmin*(1+np.exp(Om*T))
    if X is None: X=lambda t:0.0; Xd=lambda t:0.0
    # start on periodic orbit: just after right-foot placement at u=+w/2, xi = u - bmin
    s=1; urel=w/2; xi0=urel-bmin
    # COM on periodic orbit at step start: y such that over step it's symmetric: y0 = u - (u - xi0)... solve: use c,cdot with xi known and symmetric y at mid-step
    # choose y0=-(w/2)*... simple: integrate a few steps on rigid ground first to converge
    y=0.0; v=0.0; t=-20*T+phase0; tn=t; out=[]
    n=int((T_end-t)/dt)
    for k in range(n):
        if t>=tn-1e-12:
            Xt,Vt=(X(t),Xd(t)) if t>0 else (0.0,0.0)
            vel=(v-Vt) if variant=='rel' else v
            urel=(y-Xt)+vel/Om+s*bmin; s=-s; tn+=T
        Xt=X(t) if t>0 else 0.0
        u=Xt+urel; a=Om**2*(y-u)
        v+=dt*a; y+=dt*v; t+=dt
        if t>0 and (rec or True): out.append((t,y,u,y+v/Om,-m*a))
    return np.array(out),w
def equiv(fp,fb,amp=0.01,variant='rel',T=120,nph=6):
    wb=2*np.pi*fb; X=lambda t:amp*np.sin(wb*t); Xd=lambda t:amp*wb*np.cos(wb*t)
    dC=[];dM=[]
    for p in np.linspace(0,1/(2*fp),nph,endpoint=False):
        o,_=walk(T,fp,X,Xd,variant,dt=2e-3,phase0=p); t=o[:,0]; F=o[:,4]; sel=t>10
        V=amp*wb*np.cos(wb*t[sel]); A=-amp*wb**2*np.sin(wb*t[sel])
        dC.append(-np.mean(F[sel]*V)/np.mean(V*V)); dM.append(-np.mean(F[sel]*A)/np.mean(A*A))
    return np.mean(dC),np.mean(dM)
if __name__=='__main__':
    o,w=walk(5,0.9); print('Om',Om,'step width',w,'peak force',np.abs(o[:,4]).max(),'COM sway pk-pk',np.ptp(o[o[:,0]>2,1]))
    for fb in (0.5,0.7,0.9,1.1,1.3):
        print(fb,[np.round(equiv(0.9,fb,variant=v),1) for v in ('rel','abs')])
