"""Hybrid: independence Metropolis whose proposals come from a scrambled digital net (one net coordinate per
torsion, consumed in systematic-scan order) while acceptance uniforms stay pseudo-random. Also reports the
integrated autocorrelation time of the plain chain."""
import numpy as np, json
from trap import Trap, TWO_PI
from r42_trap import chain
from samplers import sobol
R=20; ms=list(range(8,17)); NMAX=2**16; out={}
acf=json.load(open("../data/r42.json"))["acf"]
for g in [0.6,0.8,1.25]:
    l=np.array(acf[str(g)]["lags"],float); a=np.array(acf[str(g)]["acf"]); tau=0.5+np.trapezoid(np.clip(a,0,None),l)
    out[f"tau_int_{g}"]=float(tau); T=Trap(g); _,Eu1,_,_=T.exact1(2**20)
    for d in [1,2,4]:
        EU=d*Eu1; rng=np.random.default_rng(7); th0=TWO_PI*rng.random((R,d)); rec=set(2**m for m in ms)
        steps=NMAX//d
        U1=np.empty((NMAX,R))
        for r in range(R):
            P=sobol(int(np.ceil(np.log2(steps))),d,1000+r)[:steps]      # (steps,d)
            U1[:,r]=P.reshape(-1)[:NMAX]                                # step k updates torsion k%d with P[k//d,k%d]
        o=chain(T,d,NMAX,U1,rng.random((NMAX,R)),th0,"imh",record=rec)
        e=np.array([o[2**m]-EU for m in ms]).T; rm=np.sqrt((e**2).mean(0))
        sel=np.array(ms)>=10; sl=float(np.polyfit(np.array(ms)[sel],np.log2(rm[sel]),1)[0])
        out[f"{g}|{d}"]={"rmse":rm.tolist(),"slope":sl}; np.save(f"../data/r42_{g}_{d}_imh_net.npy",e)
        ref=json.load(open("../data/r42.json"))["err"][f"{g}|{d}"]
        i14=ms.index(14)
        print(f"g={g} d={d}: tau_int={tau:.0f}  RMSE@2^14 imh_net {rm[i14]:.3g} vs imh_prng {ref['imh_prng'][i14]:.3g}, net {ref['net'][i14]:.3g}, rwm {ref['rwm'][i14]:.3g}; slope {sl:.2f}",flush=True)
json.dump(out,open("../data/r44.json","w"),indent=1)
