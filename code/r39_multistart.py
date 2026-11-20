"""Multistart: K short MALA chains seeded from net points vs iid points vs one long chain, same total evaluations."""
import numpy as np, json
from landscape import *
from samplers import sobol, rand
from r38_dyn import chain, tune, NB
R=20; Ntot=2**14; res={}
for d in [4,8,16]:
    for L in ["diffusive_1","diffusive_2","funnel_8"]:
        u,du=make(L); EU=exact_EU(u,d); pm=exact_marginal(u,d,d//2)
        rng=np.random.default_rng(3); s=tune("mala",u,du,d,rng); rec={}
        for K in [16,64,256]:
            n=Ntot//K
            for seed_kind in ["net","iid"]:
                eU=[];tv=[]
                for r in range(R):
                    X=sobol(int(np.log2(K)),d,r) if seed_kind=="net" else rand(int(np.log2(K)),d,500+r)
                    th0=2*np.pi*X
                    # run all K chains n steps; pool after discarding first 20% of each chain
                    th=th0.copy(); E=U(th,u); G=gradU(th,du); sumE=0; H=np.zeros(NB); cnt=0; b=int(0.2*n)
                    for k in range(n):
                        prop=th-0.5*s*G+np.sqrt(s)*rng.normal(size=th.shape); Ep=U(prop,u); Gp=gradU(prop,du)
                        fw=prop-th+0.5*s*G; bw=th-prop+0.5*s*Gp; la=E-Ep-((bw**2).sum(1)-(fw**2).sum(1))/(2*s)
                        a=np.log(rng.uniform(size=K))<la; th[a]=prop[a]%(2*np.pi); E[a]=Ep[a]; G[a]=Gp[a]
                        if k>=b:
                            sumE+=E.sum(); cnt+=K; np.add.at(H,(th[:,d//2]/(2*np.pi)*NB).astype(int)%NB,1)
                    eU.append(sumE/cnt-EU); tv.append(0.5*np.abs(H/H.sum()-pm).sum())
                rec[f"K{K}_{seed_kind}"]={"rmse_U":float(np.sqrt(np.mean(np.square(eU)))),"TV":float(np.mean(tv))}
        res[f"{L}|d{d}"]=rec
        print(L,d,{k:(round(v['rmse_U'],3),round(v['TV'],3)) for k,v in rec.items()},flush=True)
json.dump(res,open("../data/r39.json","w"),indent=1)
