"""Heavy-tailed memory test on Bouchaud trap landscapes.
(1) equilibrium autocorrelation of the energy along Metropolis chains (independence proposals);
(2) error of <U> versus number of energy evaluations for quadrature (iid, scrambled net, shifted lattice)
    and for Metropolis chains driven by pseudo-random numbers, by an LFSR (Tausworthe) CUD driver,
    and local random-walk Metropolis."""
import numpy as np, json, time
from trap import Trap, TWO_PI
from samplers import sobol, rand
from scipy.stats import qmc
R=20; GAMMAS=[0.6,0.8,1.25]; DIMS=[1,2,4]
ms=list(range(8,17)); NMAX=2**16
PRIM={11:0b100000000101,13:0b10000000011011,15:0b1000000000000011,17:(1<<17)|(1<<3)|1}
def tausworthe(m):
    """u_i = sum_j b_{i m + j} 2^{-j}, b an m-sequence of the primitive polynomial of degree m (period P=2^m-1)."""
    f=PRIM[m]; P=2**m-1; s=1; b=np.empty(P,dtype=np.int64)
    for k in range(P):
        b[k]=(s>>(m-1))&1; s<<=1
        if s>>m&1: s^=f
    idx=(np.arange(P)[:,None]*m+np.arange(m)[None,:])%P
    return (b[idx]*(0.5**np.arange(1,m+1))).sum(1)
def equilibrium_sample(T,n,d,rng):
    _,_,g,p=T.exact1(2**16); cdf=np.cumsum(p)
    return g[np.searchsorted(cdf,rng.random((n,d)))]+ (rng.random((n,d))-0.5)*TWO_PI/2**16
def chain(T,d,N,U1,U2,th0,kind="imh",step=None,record=None):
    """systematic-scan single-torsion Metropolis; U1 proposal uniforms (N,R), U2 acceptance uniforms (N,R)."""
    th=th0.copy(); R_=th.shape[0]; uc=T.u(th); E=uc.sum(1); sumE=np.zeros(R_); out={}
    for k in range(N):
        i=k%d
        new=TWO_PI*U1[k] if kind=="imh" else th[:,i]+step*np.sqrt(-2*np.log(np.maximum(U1[k],1e-300)))*np.cos(TWO_PI*U2[k]*0.999+0.0005)
        un=T.u(new); acc=(U2[k] if kind=="imh" else np.random.default_rng(k).random(R_))<np.exp(np.minimum(uc[:,i]-un,0))
        th[acc,i]=np.mod(new[acc],TWO_PI); E=E+np.where(acc,un-uc[:,i],0); uc[acc,i]=un[acc]
        sumE+=E
        if record is not None and (k+1) in record: out[k+1]=sumE/(k+1)
    return out
if __name__=='__main__':
    res={"gammas":GAMMAS,"dims":DIMS,"ms":ms,"acf":{},"err":{}}
    t0=time.time()
    # ---------- (1) autocorrelation of U along equilibrium IMH chains, d=1
    for g in GAMMAS:
        T=Trap(g); rng=np.random.default_rng(1); L=2**18; C=64
        th=equilibrium_sample(T,C,1,rng); uc=T.u(th[:,0]); trace=np.empty((L,C))
        for k in range(L):
            new=TWO_PI*rng.random(C); un=T.u(new); a=rng.random(C)<np.exp(np.minimum(uc-un,0))
            uc=np.where(a,un,uc); trace[k]=uc
        x=trace-trace.mean(); F=np.fft.rfft(x,n=2*L,axis=0); ac=np.fft.irfft(F*np.conj(F),axis=0)[:L].mean(1); ac/=ac[0]
        lags=np.unique(np.logspace(0,4.5,60).astype(int)); res["acf"][str(g)]={"lags":lags.tolist(),"acf":ac[lags].tolist()}
        sel=(lags>=10)&(lags<=min(3000,np.exp(T.eps.max()-T.eps.min())/3))&(ac[lags]>0)
        slope=np.polyfit(np.log(lags[sel]),np.log(ac[lags][sel]),1)[0] if sel.sum()>3 else np.nan
        res["acf"][str(g)]["slope"]=float(slope)
        print(f"gamma {g}: ACF slope {slope:.2f} (fit window lags 10..{lags[sel].max() if sel.any() else 0})  [{time.time()-t0:.0f}s]",flush=True)
    # ---------- (2) error of <U> vs N
    taus={m:tausworthe(m) for m in PRIM}
    for g in GAMMAS:
        T=Trap(g); _,Eu1,_,_=T.exact1(2**20)
        for d in DIMS:
            EU=d*Eu1; key=f"{g}|{d}"; r={"EU":EU}; rng=np.random.default_rng(7)
            def qerr(P):
                u=T.u(TWO_PI*P).sum(1); w=np.exp(-(u-u.min())); cs=np.cumsum(w); cu=np.cumsum(w*u)
                return np.array([cu[2**m-1]/cs[2**m-1]-EU for m in ms])
            r["iid"]=np.array([qerr(rand(16,d,300+s)) for s in range(R)])
            r["net"]=np.array([qerr(sobol(16,d,s)) for s in range(R)])
            lat=np.zeros((R,len(ms)))
            for jm,m in enumerate(ms):
                N=2**m; best=None; k=np.arange(N)
                for gg in np.random.default_rng(m).choice(np.arange(3,N,2),size=min(200,N//2-1),replace=False):
                    z=np.array([pow(int(gg),j,N) for j in range(d)]); X=(np.outer(k,z)%N)/N
                    wd=-(4/3)**d+np.mean(np.prod(1.5-X*(1-X),axis=1))
                    if best is None or wd<best[0]: best=(wd,X)
                for s in range(R):
                    P=(best[1]+rng.random(d))%1; u=T.u(TWO_PI*P).sum(1); w=np.exp(-(u-u.min())); lat[s,jm]=(w*u).sum()/w.sum()-EU
            r["lattice"]=lat
            th0=TWO_PI*rng.random((R,d)); rec=set(2**m for m in ms)
            o=chain(T,d,NMAX,rng.random((NMAX,R)),rng.random((NMAX,R)),th0,"imh",record=rec)
            r["imh_prng"]=np.array([o[2**m]-EU for m in ms]).T
            # CUD (Tausworthe) driver: full period per run, random Cranley-Patterson shifts per replica
            cud=np.full((R,len(ms)),np.nan)
            for m,uu in taus.items():
                N=(len(uu)-1)//2; sh=rng.random((2,R))
                U1=(uu[0:2*N:2][:,None]+sh[0])%1; U2=(uu[1:2*N:2][:,None]+sh[1])%1
                o=chain(T,d,N,U1,U2,th0,"imh",record={N})
                jm=int(np.argmin([abs(N-2**mm) for mm in ms])); cud[:,jm]=o[N]-EU
            r["imh_cud"]=cud
            o=chain(T,d,NMAX,rng.random((NMAX,R)),rng.random((NMAX,R)),th0,"rwm",step=T.h/4,record=rec)
            r["rwm"]=np.array([o[2**m]-EU for m in ms]).T
            out={"EU":EU}
            for k,v in r.items():
                if k=="EU": continue
                rm=np.sqrt(np.nanmean(v**2,0)); out[k]=rm.tolist()
                ok=~np.isnan(rm); x=np.log2(np.array([2**m for m in ms]))[ok]; y=np.log2(rm[ok])
                sel=x>=10; out[k+"_slope"]=float(np.polyfit(x[sel],y[sel],1)[0]) if sel.sum()>=2 else None
                np.save(f"../data/r42_{g}_{d}_{k}.npy",v)
            res["err"][key]=out
            i14=ms.index(14)
            print(f"g={g} d={d}: RMSE@2^14 "+" ".join(f"{k} {out[k][i14]:.3g}" for k in ["iid","net","lattice","imh_prng","imh_cud","rwm"])
                  +" | slopes "+" ".join(f"{k} {out[k+'_slope']:.2f}" for k in ["iid","net","lattice","imh_prng","imh_cud","rwm"] if out[k+'_slope'] is not None)+f"  [{time.time()-t0:.0f}s]",flush=True)
    json.dump(res,open("../data/r42.json","w"))
