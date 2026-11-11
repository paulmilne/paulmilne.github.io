import numpy as np, json
from scipy.optimize import fsolve
g=0.5
def fold(w0):
    guess=None; Ks=[]
    for r in np.linspace(0.002,0.9,220):
        def eq(u):
            rho,phi,K=u; z1=rho*np.exp(1j*phi); d=(1j*w0-g)*z1+K/2*(r-r*z1**2); return [d.real,d.imag,rho*np.cos(phi)-r]
        if guess is None:
            K0=2*(g*g+w0*w0)/g
            for phi0 in np.linspace(-1.5,1.5,61):
                u,_,ier,_=fsolve(eq,[r/np.cos(phi0),phi0,K0],full_output=True)
                if ier==1 and 0<u[0]<=1 and abs(u[2]-K0)<0.05: guess=u; break
        u,_,ier,_=fsolve(eq,guess,full_output=True)
        if ier==1: guess=u; Ks.append(u[2])
    return min(Ks)
xs=np.linspace(0.02,2.2,56); f=[fold(x*g)/g for x in xs]
json.dump({'x':xs.tolist(),'Ksn':f},open('fold.json','w')); print(list(zip(xs.round(2),np.round(f,3)))[::5])
