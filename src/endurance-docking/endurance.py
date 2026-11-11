"""Endurance mass model, damage, tumble and docking requirements (assumed masses; see table in the post)."""
import numpy as np
g0=9.81
R=29.0                     # module centre radius (m); ring outer diameter ~64 m
BOX=(10.0,8.0,8.0)         # module size: tangential, radial, axial (m)
TYPES={'engine':50e3,'cargo':40e3,'habitat':30e3,'cryo':30e3,'command':30e3}
LAYOUT=['engine','cargo','habitat','engine','cargo','habitat','engine','cargo','cryo','engine','cargo','command']
HUB=40e3                   # hub, spokes, docking collar
LOST=[0,1]                 # engine at 0 deg and the cargo module next to it
def modules(lost=()):
    out=[]
    for k,t in enumerate(LAYOUT):
        if k in lost: continue
        a=np.radians(30*k); out.append((t,TYPES[t],np.array([R*np.cos(a),R*np.sin(a),0.0]),a))
    return out
def mass_props(lost=()):
    mods=modules(lost); M=HUB+sum(m for _,m,_,_ in mods)
    com=sum(m*p for _,m,p,_ in mods)/M
    I=np.zeros((3,3))
    def add_point(m,p):
        r=p-com; I[:]+=m*(np.dot(r,r)*np.eye(3)-np.outer(r,r))
    add_point(HUB,np.zeros(3)); I[:]+=HUB*np.diag([1,1,2])*4.0**2/4   # hub: disc radius 4 m
    a,b,c=BOX
    for t,m,p,ang in mods:
        add_point(m,p)
        # box inertia in its own axes (tangential, radial, axial) rotated to ring frame
        Ib=m/12*np.diag([b*b+c*c,a*a+c*c,a*a+b*b])
        Rz=np.array([[-np.sin(ang),np.cos(ang),0],[np.cos(ang),np.sin(ang),0],[0,0,1]]).T
        I[:]+=Rz@Ib@Rz.T
    return M,com,I,mods
def rpm2w(r): return r*2*np.pi/60
if __name__=='__main__':
    for lost in ((),LOST):
        M,com,I,_=mass_props(lost); ev,V=np.linalg.eigh(I)
        print('lost',lost,'M t',M/1e3,'COM',com.round(2),'|d|',np.linalg.norm(com).round(2),'principal 1e8',(ev/1e8).round(3))
        print('  axis of max',V[:,2].round(3),'in-plane principal angle deg',np.degrees(np.arctan2(V[1,0],V[0,0])).round(1))
    M,com,I,_=mass_props(LOST); d=np.linalg.norm(com); I3=np.linalg.eigvalsh(I)[2]
    for r in (5.5,20,68):
        w=rpm2w(r)
        print(f'{r} rpm: ring g {w*w*R/g0:.2f}, engine module pull {50e3*w*w*R/g0/1e3:.0f} t-f, port speed {d*w:.1f} m/s, port accel {d*w*w/g0:.2f} g, L {I3*w:.3g}, KE {0.5*I3*w*w/4.184e9:.2f} t TNT, impulse at R {I3*w/R:.3g} N s, dv {I3*w/R/M:.0f} m/s')
