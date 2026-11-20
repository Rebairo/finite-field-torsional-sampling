"""Direct test of the long-memory variance law: equilibrium-started independence-Metropolis chains on the
1D trap landscape; RMSE of the time average of U versus N, compared with N^(-alpha/2) (alpha from the ACF)."""
import numpy as np, json
from trap import Trap, TWO_PI
from r42_trap import equilibrium_sample
out={}
for g in [0.6,0.8,1.25]:
    T=Trap(g); _,EU,_,_=T.exact1(2**20); rng=np.random.default_rng(11); C=2000; L=2**16
    th=equilibrium_sample(T,C,1,rng)[:,0]; uc=T.u(th); cs=np.zeros(C); Ns=2**np.arange(2,17); rec={}
    for k in range(1,L+1):
        new=TWO_PI*rng.random(C); un=T.u(new); a=rng.random(C)<np.exp(np.minimum(uc-un,0)); uc=np.where(a,un,uc); cs+=uc
        if k in set(Ns): rec[k]=np.sqrt(np.mean((cs/k-EU)**2))
    y=np.array([rec[n] for n in Ns]); cut=np.exp(T.eps.max()-T.eps.min())
    lo=Ns<=cut/4; hi=Ns>=cut*4
    s_lo=np.polyfit(np.log(Ns[lo]),np.log(y[lo]),1)[0] if lo.sum()>2 else None
    s_hi=np.polyfit(np.log(Ns[hi]),np.log(y[hi]),1)[0] if hi.sum()>2 else None
    out[str(g)]={"N":Ns.tolist(),"rmse":y.tolist(),"cutoff":float(cut),"slope_below":s_lo,"slope_above":s_hi}
    print(g,"cutoff",round(cut),"slope below cutoff",s_lo,"above",s_hi)
json.dump(out,open("../data/r43.json","w"),indent=1)
