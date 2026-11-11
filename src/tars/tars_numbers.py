import math, numpy as np, mujoco
from tars import *
from walk import run
GM=1.3*G
l=LS/2-RF+RF; M0=4*MS+4.4; I0=4*MS*(LS**2+D**2)/12
def k2(M,I=I0): return I/(M*l*l)
def roll_power(v,M=M0,n=8,g=GM):
    a=math.pi/n; K=k2(M); r=ratio(n,K); J=M*l*l*(1+K)
    w=v*a/(l*math.sin(a)); loss=0.5*J*w*w*(1-r*r); return loss*w/(2*a), r, w
def drag_power(v,depth=0.5,width=1.2,Cd=1.0): return 0.5*1000*Cd*width*depth*v**3
for M,lab in ((M0,'alone'),(M0+80,'carrying Brand (80 kg at hub)')):
    K=k2(M); print(lab,'k2 %.3f'%K,'ratio 8: %.3f  ratio 4: %.3f'%(ratio(8,K),ratio(4,K)),'energy kept 8: %.2f 4: %.2f'%(ratio(8,K)**2,ratio(4,K)**2))
    for v in (2,5,10):
        P,r,w=roll_power(v,M); print(f'  v {v}: impacts {P/1e3:.1f} kW, water {drag_power(v)/1e3:.1f} kW, omega {w:.1f}, liftoff limit omega {math.sqrt(GM/l):.2f}')
print('walk Froude 0.5 speed earth %.2f miller %.2f; Fr=1: %.2f %.2f'%(math.sqrt(0.5*G*l),math.sqrt(0.5*GM*l),math.sqrt(G*l),math.sqrt(GM*l)))
print('rolling hub-on-ground limit v (8 spoke):', math.sqrt(GM/l)*l*math.sin(math.pi/8)/(math.pi/8))
# walking: actuator work
m=None
st,fr,d=run(15,0.02,T=20,record=False)
