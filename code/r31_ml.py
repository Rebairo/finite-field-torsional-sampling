"""3.1  Mittag-Leffler relaxation as a sum of exponentials.
E_g(-t^g) = int_0^inf e^{-rt} K_g(r) dr,  K_g(r) = sin(g pi)/pi * r^{g-1}/(r^{2g}+2 r^g cos(g pi)+1)  (complete monotonicity)."""
import numpy as np, json
from scipy.integrate import quad
from scipy.optimize import nnls
from scipy.special import erfcx, gamma as G

def K(r,g): return np.sin(g*np.pi)/np.pi * r**(g-1)/(r**(2*g)+2*r**g*np.cos(g*np.pi)+1)
def ML(t,g):
    # split integral in log space for accuracy
    f=lambda u: K(np.exp(u),g)*np.exp(u)*np.exp(-np.exp(u)*t)
    return quad(f,-60,np.log(60/t)+5,limit=400,epsabs=0,epsrel=1e-11)[0]

# validation against closed form at g=1/2: E_{1/2}(-sqrt t) = erfcx(sqrt t)
tv=np.logspace(-2,4,25)
err=max(abs(ML(t,0.5)-erfcx(np.sqrt(t)))/erfcx(np.sqrt(t)) for t in tv)
print("validation g=0.5 max rel err", err)

out={"validation_rel_err":err,"fits":{},"ndec":{}}
gs=[0.3,0.5,0.7,0.9]
t=np.logspace(-2,4,300)              # six decades
curves={}
for g in gs:
    y=np.array([ML(x,g) for x in t]); curves[g]=y
    # asymptotic tail check
    tail=t[-1]**(-g)/G(1-g)
    out.setdefault("tail_ratio",{})[g]=y[-1]/tail
    res=[]
    for n in [2,4,6,8,10,12,16,20,24]:
        r=np.logspace(-2-np.log10(t[-1])-1, 2+1, n)   # rates span the time window +1 decade each side
        A=np.exp(-np.outer(t,r))/y[:,None]
        w,_=nnls(A,np.ones_like(t))
        e=np.max(np.abs(A@w-1)); res.append((n,e))
    out["fits"][g]=res
    print("g",g,[f"{n}:{e:.1e}" for n,e in res])
# n needed for 1% max error vs number of decades (g=0.5, exact erfcx for speed)
for g in [0.5]:
    for D in [2,4,6,8,10]:
        tt=np.logspace(-1,-1+D,200); yy=erfcx(np.sqrt(tt))
        for n in range(2,60):
            r=np.logspace(-(-1+D)-1,2,n); A=np.exp(-np.outer(tt,r))/yy[:,None]
            w,_=nnls(A,np.ones_like(tt))
            if np.max(np.abs(A@w-1))<0.01: break
        out["ndec"][D]=n; print("decades",D,"n for 1%:",n)
np.savez("../data/r31_curves.npz",t=t,**{f"g{g}":curves[g] for g in gs})
json.dump(out,open("../data/r31.json","w"),indent=1,default=float)
