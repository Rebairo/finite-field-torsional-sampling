import numpy as np, json, warnings
from scipy.stats import qmc
from samplers import *
warnings.filterwarnings("ignore")
R=10; out=[]
for d in [2,4,8,16]:
    for m in range(6,13):
        rec={"d":d,"m":m}
        for meth in ["WD","L2-star"]:
            rec[f"sobol_{meth}"]=qmc.discrepancy(sobol(m,d),method=meth)
            rec[f"symshift_{meth}"]=qmc.discrepancy(sym_shift(m,d),method=meth)
            if d<=m: rec[f"symfrob_{meth}"]=qmc.discrepancy(sym_frob(m,d),method=meth)
            for name,fn in [("scr",lambda s: sobol(m,d,s)),("rand",lambda s: rand(m,d,s))]:
                v=[qmc.discrepancy(fn(s),method=meth) for s in range(R)]
                rec[f"{name}_{meth}"]=float(np.mean(v)); rec[f"{name}_{meth}_sd"]=float(np.std(v))
        # 1D projections: exact BAM grid?
        rec["sobol_1D_exact"]=bool(all(np.array_equal(np.sort(sobol(m,d)[:,j]),np.arange(2**m)/2**m) for j in range(d)))
        rec["symshift_1D_exact"]=bool(all(np.array_equal(np.sort(sym_shift(m,d)[:,j]),np.arange(2**m)/2**m) for j in range(d)))
        out.append(rec); print({k:(round(v,6) if isinstance(v,float) else v) for k,v in rec.items() if 'sd' not in k})
json.dump(out,open("../data/r34_disc.json","w"),indent=1)
