"""3.6  Convergence of thermodynamic averages on chain landscapes over the torus T^d.
U(theta) = sum_i u(theta_i) + c * sum_i cos(theta_i - theta_{i+1})   (energies in k_BT)
Exact Z and <U> from a transfer matrix on a fine periodic grid."""
import numpy as np, json, warnings
from samplers import sobol, rand, sym_shift
warnings.filterwarnings("ignore")
rng0=np.random.default_rng(7); K=4
amp=rng0.normal(size=K)/np.arange(1,K+1); ph=rng0.uniform(0,2*np.pi,K)
def u_rough(th,eps):
    v=sum(amp[k]*np.cos((k+1)*th+ph[k]) for k in range(K))
    g=np.linspace(0,2*np.pi,4096,endpoint=False); s=sum(amp[k]*np.cos((k+1)*g+ph[k]) for k in range(K)).std()
    return eps*v/s
LAND={"diffusive_0.5":lambda t:u_rough(t,0.5),"diffusive_1":lambda t:u_rough(t,1.0),
      "diffusive_2":lambda t:u_rough(t,2.0),"rugged_4":lambda t:u_rough(t,4.0),
      "funnel_8":lambda t:-8*np.cos(t)}
C=0.5
def U(X,uf):   # X in [0,1)^d
    th=2*np.pi*X; return uf(th).sum(1)+C*np.cos(th[:,:-1]-th[:,1:]).sum(1)
def exact(uf,d,beta=1.0,M=1024):
    g=np.linspace(0,2*np.pi,M,endpoint=False); w=1.0/M        # normalised measure
    a=np.exp(-beta*uf(g)); T=np.exp(-beta*C*np.cos(g[:,None]-g[None,:]))
    v=a*w; lz=0.0
    for _ in range(d-1):
        v=(v@T)*a*w; s=v.sum(); lz+=np.log(s); v/=s
    return lz+np.log(v.sum())
def exact_EU(uf,d):
    h=1e-4; return -(exact(uf,d,1+h)-exact(uf,d,1-h))/(2*h)
R=20; mmax=16; ms=list(range(6,mmax+1))
res={}
for d in [4,8,16,24]:
    for name,uf in LAND.items():
        lz=exact(uf,d); Z=np.exp(lz); EU=exact_EU(uf,d)
        rec={"lnZ":lz,"EU":EU}
        def errs(P):
            E=U(P,uf); f=np.exp(-(E-E.min()))*np.exp(-E.min())   # stable
            cs=np.cumsum(f); csE=np.cumsum(f*E); out=[]
            for m in ms:
                n=2**m; out.append((cs[n-1]/n/Z-1, csE[n-1]/cs[n-1]-EU, np.log(cs[n-1]/n)-lz))
            return np.array(out)
        er={"scr":np.array([errs(sobol(mmax,d,s)) for s in range(R)]),
            "rand":np.array([errs(rand(mmax,d,100+s)) for s in range(R)])}
        for k,v in er.items():
            rec[k+"_Z"]=np.sqrt((v[:,:,0]**2).mean(0)).tolist(); rec[k+"_U"]=np.sqrt((v[:,:,1]**2).mean(0)).tolist(); rec[k+"_lnZ"]=np.sqrt((v[:,:,2]**2).mean(0)).tolist()
        for k,P in [("sobol",sobol(mmax,d)),("sym",sym_shift(mmax,d) if d<=mmax else None)]:
            if P is None: continue
            v=errs(P); rec[k+"_Z"]=np.abs(v[:,0]).tolist(); rec[k+"_U"]=np.abs(v[:,1]).tolist(); rec[k+"_lnZ"]=np.abs(v[:,2]).tolist()
        # relative first-order variation proxy of the Boltzmann factor
        P=sobol(16,d,999); th=2*np.pi*P; E=U(P,uf); f=np.exp(-(E-E.min()))
        h=1e-5; grad=[]
        for i in range(d):
            Q=P.copy(); Q[:,i]=(Q[:,i]+h)%1; grad.append(np.abs(np.exp(-(U(Q,uf)-E.min()))-f)/h)
        rec["ESS_frac"]=float(f.sum()**2/(len(f)*(f**2).sum()))
        rec["Vproxy"]=float(np.sum([g.mean() for g in grad])/f.mean())
        sl=lambda y: float(np.polyfit(np.log2([2**m for m in ms[3:]]),np.log2(y[3:]),1)[0])
        rec["slope_scr_Z"]=sl(rec["scr_Z"]); rec["slope_rand_Z"]=sl(rec["rand_Z"])
        rec["slope_scr_lnZ"]=sl(rec["scr_lnZ"]); rec["slope_rand_lnZ"]=sl(rec["rand_lnZ"])
        rec["gain_lnZ_N2^14"]=rec["rand_lnZ"][ms.index(14)]/rec["scr_lnZ"][ms.index(14)]
        rec["gain_Z_N2^14"]=rec["rand_Z"][ms.index(14)]/rec["scr_Z"][ms.index(14)]
        rec["gain_U_N2^14"]=rec["rand_U"][ms.index(14)]/rec["scr_U"][ms.index(14)]
        res[f"{name}|d{d}"]=rec
        print(f"d={d:2d} {name:14s} ESS/N={rec['ESS_frac']:.2e} V={rec['Vproxy']:7.1f} lnZ err@2^14 scr {rec['scr_lnZ'][8]:.3f} rand {rec['rand_lnZ'][8]:.3f} gain {rec['gain_lnZ_N2^14']:5.1f} slopes {rec['slope_scr_lnZ']:.2f}/{rec['slope_rand_lnZ']:.2f} | <U> err scr {rec['scr_U'][8]:.3f} rand {rec['rand_U'][8]:.3f}")
json.dump({"ms":ms,"C":C,"res":res},open("../data/r36.json","w"))
