"""Compare quadrature (iid, scrambled net, shifted Korobov lattice) with Markov-chain samplers (RWM, MALA)
on chain landscapes. Cost = number of energy (or energy+gradient) evaluations N."""
import numpy as np, json, time
from landscape import *
from samplers import sobol, rand
R=20; mmax=16; ms=list(range(6,mmax+1)); NB=36
LANDS=["diffusive_0.5","diffusive_1","diffusive_2","funnel_8"]
def korobov_g(N,d,ncand=300,seed=0):
    rng=np.random.default_rng(seed); best=None
    k=np.arange(N)
    for g in rng.choice(np.arange(3,N,2),size=min(ncand,N//2-1),replace=False):
        z=np.array([pow(int(g),j,N) for j in range(d)])
        X=(np.outer(k,z)%N)/N
        wd=-(4/3)**d+np.mean(np.prod(1.5-X*(1-X),axis=1))
        if best is None or wd<best[0]: best=(wd,z)
    return best[1]
def lattice(N,d,z,shift): return ((np.outer(np.arange(N),z)%N)/N+shift)%1
def quad_stats(X,u,d,lz,EU,pm):
    th=2*np.pi*X; E=U(th,u); f=np.exp(-(E-E.min())); lzh=np.log(f.mean())-E.min()
    Eu=(f*E).sum()/f.sum(); h=np.bincount((th[:,d//2]/(2*np.pi)*NB).astype(int)%NB,weights=f,minlength=NB); h/=h.sum()
    return lzh-lz, Eu-EU, 0.5*np.abs(h-pm).sum()
def quad_prefix(X,u,d,lz,EU,pm):
    th=2*np.pi*X; E=U(th,u); f=np.exp(-(E-E.min())); bins=(th[:,d//2]/(2*np.pi)*NB).astype(int)%NB
    out=[]
    for m in ms:
        n=2**m; ff=f[:n]; lzh=np.log(ff.mean())-E.min(); Eu=(ff*E[:n]).sum()/ff.sum()
        h=np.bincount(bins[:n],weights=ff,minlength=NB); h/=h.sum(); out.append((lzh-lz,Eu-EU,0.5*np.abs(h-pm).sum()))
    return np.array(out)
def tune(kind,u,du,d,rng):
    th=rng.uniform(0,2*np.pi,(8,d)); target=0.25 if kind=="rwm" else 0.57; s=0.5 if kind=="rwm" else 0.05
    for it in range(40):
        acc=chain(kind,u,du,d,th,s,300,rng,record=False)
        s*=np.exp(1.5*(acc-target))
    return s
def chain(kind,u,du,d,th0,s,n,rng,record=True,pm=None):
    th=th0.copy(); E=U(th,u); nc=th.shape[0]
    if kind=="mala": G=gradU(th,du)
    acc=0; sumE=np.zeros(nc); H=np.zeros((nc,NB)); out=[]
    rec=set(2**m for m in ms)
    for k in range(1,n+1):
        if kind=="rwm":
            prop=th+s*rng.normal(size=th.shape); Ep=U(prop,u); la=E-Ep
        else:
            prop=th-0.5*s*G+np.sqrt(s)*rng.normal(size=th.shape); Ep=U(prop,u); Gp=gradU(prop,du)
            fw=prop-th+0.5*s*G; bw=th-prop+0.5*s*Gp
            la=E-Ep-((bw**2).sum(1)-(fw**2).sum(1))/(2*s)
        a=np.log(rng.uniform(size=nc))<la
        th[a]=prop[a]%(2*np.pi); E[a]=Ep[a]; acc+=a.mean()
        if kind=="mala": G[a]=Gp[a]
        if record:
            sumE+=E; H[np.arange(nc),(th[:,d//2]/(2*np.pi)*NB).astype(int)%NB]+=1
            if k in rec:
                Hn=H/k; out.append(np.stack([np.full(nc,np.nan),sumE/k,0.5*np.abs(Hn-pm).sum(1)],1))
    return (np.array(out).transpose(1,0,2) if record else acc/n)
if __name__=="__main__":
    res={}
    t0=time.time()
    for d in [4,8,16]:
        for L in LANDS:
            u,du=make(L); lz=exact(u,d); EU=exact_EU(u,d); pm=exact_marginal(u,d,d//2)
            key=f"{L}|d{d}"; r={"lnZ":lz,"EU":EU}
            errs={}
            errs["iid"]=np.array([quad_prefix(rand(mmax,d,100+s),u,d,lz,EU,pm) for s in range(R)])
            errs["net"]=np.array([quad_prefix(sobol(mmax,d,s),u,d,lz,EU,pm) for s in range(R)])
            lat=np.zeros((R,len(ms),3)); rng=np.random.default_rng(5)
            for i,m in enumerate(ms):
                z=korobov_g(2**m,d,seed=m); r.setdefault("korobov_z",{})[m]=z.tolist()
                for s in range(R): lat[s,i]=quad_stats(lattice(2**m,d,z,rng.uniform(size=d)),u,d,lz,EU,pm)
            errs["lattice"]=lat
            for kind in ["rwm","mala"]:
                rng=np.random.default_rng(11); s=tune(kind,u,du,d,rng); r[kind+"_step"]=float(s)
                errs[kind]=chain(kind,u,du,d,rng.uniform(0,2*np.pi,(R,d)),s,2**mmax,rng,pm=pm); errs[kind][:,:,1]-=EU
            for k,v in errs.items():
                np.save(f"../data/r38_{key.replace('|','_')}_{k}.npy",v)
                r[k]={"rmse_lnZ":np.sqrt(np.nanmean(v[:,:,0]**2,0)).tolist() if not np.all(np.isnan(v[:,:,0])) else None,
                      "rmse_U":np.sqrt(np.mean(v[:,:,1]**2,0)).tolist(),"mean_TV":np.mean(v[:,:,2],0).tolist()}
            res[key]=r
            i14=ms.index(14)
            print(f"{key:20s} |<U> err@2^14: "+"  ".join(f"{k} {r[k]['rmse_U'][i14]:.3f}" for k in errs)+" | TV: "+" ".join(f"{k} {r[k]['mean_TV'][i14]:.3f}" for k in errs)+f"  ({time.time()-t0:.0f}s)",flush=True)
    json.dump({"ms":ms,"res":res},open("../data/r38.json","w"))
