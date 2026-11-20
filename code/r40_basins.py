"""DNCS-like multistart minimisation: distinct minima found from K net starts vs K iid starts."""
import numpy as np, json
from landscape import *
from samplers import sobol, rand
def minimise(th,du,steps=3000,lr=0.02):
    th=th.copy()
    for k in range(steps):
        g=gradU(th,du); th-=lr*g
    return th%(2*np.pi), np.abs(gradU(th,du)).max(1)
def keys(P): return set(map(tuple,np.round(((P+0.05)%(2*np.pi))/0.1).astype(int)))
res={}
for d in [4,8]:
    for L in ["diffusive_2","rugged_4"]:
        u,du=make(L)
        # reference set of minima from a large pool
        big=np.vstack([2*np.pi*rand(13,d,999),2*np.pi*sobol(13,d,999)])
        mb,gb=minimise(big,du); mb=mb[gb<1e-4]
        K0=keys(mb); kE={}
        for p,e in zip(mb,U(mb,u)): kE[tuple(np.round(((p+0.05)%(2*np.pi))/0.1).astype(int))]=e
        Emin=min(kE.values()); gkey=min(kE,key=kE.get)
        rec={"n_ref_minima":len(K0)}
        for m in [4,6,8,10]:
            for kind in ["net","iid"]:
                found=[];wt=[];glob=[]
                for r in range(20):
                    X=2*np.pi*(sobol(m,d,r) if kind=="net" else rand(m,d,700+r))
                    mm,gg=minimise(X,du); mm=mm[gg<1e-4]; f=keys(mm)
                    found.append(len(f)); glob.append(gkey in f)
                    Z=sum(np.exp(-(e-Emin)) for e in kE.values()); wt.append(sum(np.exp(-(kE[k]-Emin)) for k in f if k in kE)/Z)
                rec[f"K{2**m}_{kind}"]={"minima":float(np.mean(found)),"minima_sd":float(np.std(found)),"boltz_weight":float(np.mean(wt)),"p_global":float(np.mean(glob))}
        res[f"{L}|d{d}"]=rec
        print(L,d,"ref minima",len(K0),{k:(round(v['minima'],1),round(v['boltz_weight'],3),v['p_global']) for k,v in rec.items() if k!='n_ref_minima'},flush=True)
json.dump(res,open("../data/r40.json","w"),indent=1)
