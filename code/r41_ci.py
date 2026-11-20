import numpy as np, json
ms=list(range(6,17)); i14=ms.index(14); rng=np.random.default_rng(0); B=4000
def load(key,k): return np.load(f"../data/r38_{key}_{k}.npy")
def ratio_ci(a,b,col):   # RMSE(a)/RMSE(b)
    a=a[:,i14,col]; b=b[:,i14,col]; r=[]
    for _ in range(B):
        ia=rng.integers(0,len(a),len(a)); ib=rng.integers(0,len(b),len(b))
        r.append(np.sqrt(np.mean(a[ia]**2))/np.sqrt(np.mean(b[ib]**2)))
    return float(np.sqrt(np.mean(a**2))/np.sqrt(np.mean(b**2))), [float(x) for x in np.percentile(r,[2.5,97.5])]
def tv_ratio(a,b):
    a=a[:,i14,2]; b=b[:,i14,2]; r=[a[rng.integers(0,20,20)].mean()/b[rng.integers(0,20,20)].mean() for _ in range(B)]
    return float(a.mean()/b.mean()), [float(x) for x in np.percentile(r,[2.5,97.5])]
out={}
for d in [4,8,16]:
    for L in ["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]:
        key=f"{L}_d{d}"; A={k:load(key,k) for k in ["iid","net","lattice","rwm","mala"]}
        out[key]={"lnZ_iid/net":ratio_ci(A["iid"],A["net"],0),"lnZ_iid/lat":ratio_ci(A["iid"],A["lattice"],0),
                  "U_iid/net":ratio_ci(A["iid"],A["net"],1),"U_mala/net":ratio_ci(A["mala"],A["net"],1),
                  "U_mala/lat":ratio_ci(A["mala"],A["lattice"],1),"TV_mala/net":tv_ratio(A["mala"],A["net"]),"TV_mala/lat":tv_ratio(A["mala"],A["lattice"])}
        o=out[key]; f=lambda t:f"{t[0]:.2f} [{t[1][0]:.2f},{t[1][1]:.2f}]"
        print(f"{key:18s} lnZ iid/net {f(o['lnZ_iid/net'])} | <U> mala/net {f(o['U_mala/net'])} mala/lat {f(o['U_mala/lat'])} | TV mala/net {f(o['TV_mala/net'])}")
json.dump(out,open("../data/r41_ci.json","w"),indent=1)
