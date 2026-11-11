import numpy as np, sys, json
from ipm import *
def equiv2(fp,fb,amp,variant,T=200,nph=12):
    wb=2*np.pi*fb; X=lambda t:amp*np.sin(wb*t); Xd=lambda t:amp*wb*np.cos(wb*t); dC=[];dM=[]
    for p in np.linspace(0,1/fp,nph,endpoint=False):
        o,_=walk(T,fp,X,Xd,variant,dt=2e-3,phase0=p); t=o[:,0]; F=o[:,4]; sel=t>20
        V=amp*wb*np.cos(wb*t[sel]); A=-amp*wb**2*np.sin(wb*t[sel])
        dC.append(-np.mean(F[sel]*V)/np.mean(V*V)); dM.append(-np.mean(F[sel]*A)/np.mean(A*A))
    return float(np.mean(dC)),float(np.std(dC)),float(np.mean(dM))
if __name__=="__main__":
  variant=sys.argv[1]; fbs=np.round(np.arange(0.3,1.61,0.05),3)
  res=[(float(fb),)+equiv2(0.9,fb,0.01,variant) for fb in fbs]
  json.dump(res,open(f'curve_{variant}.json','w')); print(variant,[(r[0],round(r[1]),round(r[2])) for r in res])
