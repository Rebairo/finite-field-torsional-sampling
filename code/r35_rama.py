import numpy as np, json, warnings
from samplers import *
warnings.filterwarnings("ignore")
out={}
# --- 2D t-values / empty cells for (phi,psi) pairs, N=1024, 10 residues -> d=20
m=10; d=20; N=2**m
def tval(P,m):
    X=(P*2**m).astype(np.int64)
    for t in range(m+1):
        ok=True
        for a in range(m-t+1):
            b=m-t-a; idx=(X[:,0]>>(m-a))*(2**b)+(X[:,1]>>(m-b))
            if not np.all(np.bincount(idx,minlength=2**(m-t))==2**t): ok=False; break
        if ok: return t
def empty_frac(P,k=5):
    X=(P*2**k).astype(int); c=np.bincount(X[:,0]*2**k+X[:,1],minlength=4**k); return float((c==0).mean())
S=sobol(m,d); Sc=sobol(m,d,1); Sy=sym_shift(m,d); Ra=rand(m,d,1)
out["t_sobol_pairs"]=[tval(S[:,[2*i,2*i+1]],m) for i in range(10)]
out["t_scr_pairs"]=[tval(Sc[:,[2*i,2*i+1]],m) for i in range(10)]
out["t_sym_pairs"]=[tval(Sy[:,[2*i,2*i+1]],m) for i in range(10)]
for nm,P in [("sobol",S),("scr",Sc),("sym",Sy),("rand",Ra)]:
    out[f"empty32_{nm}"]=[empty_frac(P[:,[2*i,2*i+1]]) for i in range(10)]
print({k:v for k,v in out.items()})
# --- dyadic arc coverage vs N (1D)
cov={}
X=sobol(14,1)[:,0]; Rr=rand(14,1,2)[:,0]
for k in [2,4,6,8]:
    Ns=np.unique(np.logspace(0,14,60,base=2).astype(int))
    cov[k]={"N":Ns.tolist(),
            "sobol":[len(np.unique((X[:n]*2**k).astype(int)))/2**k for n in Ns],
            "rand":[len(np.unique((Rr[:n]*2**k).astype(int)))/2**k for n in Ns]}
out["arc_coverage"]=cov
# N needed to cover all 90-degree arcs (k=2) and 11.25 (k=5)
for k in [2,5]:
    out[f"N_full_k{k}_sobol"]=int(next(n for n in range(1,2**14) if len(np.unique((X[:n]*2**k).astype(int)))==2**k))
    out[f"N_full_k{k}_rand_mean"]=float(np.mean([next(n for n in range(1,2**14) if len(np.unique((r[:n]*2**k).astype(int)))==2**k) for r in [rand(14,1,s)[:,0] for s in range(50)]]))
print({k:v for k,v in out.items() if k.startswith("N_full")})
# --- consecutive jumps on the circle
def jumps(P): J=np.abs(((np.diff(P*360,axis=0))+180)%360-180).ravel(); return J
Js=jumps(sobol(14,6)); Jr=jumps(rand(14,6,3))
out["jump_sobol"]={"median":float(np.median(Js)),"gt90":float((Js>90).mean()),"lt10":float((Js<10).mean())}
out["jump_rand"]={"median":float(np.median(Jr)),"gt90":float((Jr>90).mean()),"lt10":float((Jr<10).mean())}
np.savez("../data/r35_jumps.npz",sobol=Js,rand=Jr)
print(out["jump_sobol"],out["jump_rand"])
# --- model Ramachandran density (von Mises mixture; generic non-Gly residue, illustrative)
basins=[("aR",-63,-43,0.40,9,9),("beta",-120,130,0.25,5,5),("PPII",-65,145,0.25,9,7),("aL",60,45,0.05,9,9),("bridge",-90,0,0.05,6,6)]
g=np.linspace(-180,180,721)[:-1]; F,Ps=np.meshgrid(g,g,indexing="ij")
def vm(x,mu,k): return np.exp(k*(np.cos(np.radians(x-mu))-1))
rho=sum(w*vm(F,a,k1)*vm(Ps,b,k2) for _,a,b,w,k1,k2 in basins); rho/=rho.sum()
o=np.sort(rho.ravel())[::-1]; cs=np.cumsum(o)
for q in [0.90,0.99]:
    thr=o[np.searchsorted(cs,q)]; out[f"vonmises_allowed_area_{int(q*100)}"]=float((rho>=thr).mean())
np.save("../data/r35_rama_rho.npy",rho)
# --- model Ramachandran distribution used in the paper (rama_model.py)
from rama_model import density
rho_m=density(F,Ps); rho_m/=rho_m.sum(); om=np.sort(rho_m.ravel())[::-1]; cm=np.cumsum(om)
for q in [0.90,0.99]: out[f"model_area_{int(q*100)}"]=float((rho_m>=om[np.searchsorted(cm,q)]).mean())
np.save("../data/r35_rama_rho_model.npy",rho_m)
json.dump(out,open("../data/r35.json","w"),indent=1)
print({k:v for k,v in out.items() if "area" in k})
