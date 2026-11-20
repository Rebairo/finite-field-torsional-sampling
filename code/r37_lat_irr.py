"""(a) Korobov lattice wrap-around discrepancy; (b) Sobol-type nets from primitive vs irreducible non-primitive polynomials."""
import numpy as np, json, warnings
from scipy.stats import qmc
from polyutil import is_primitive_gf2
warnings.filterwarnings("ignore")
out={"lattice":[], "irr":{}}
def korobov(N,d,ncand=400,seed=0):
    rng=np.random.default_rng(seed); k=np.arange(N); best=None
    for g in rng.choice(np.arange(3,N,2),size=min(ncand,N//2-1),replace=False):
        z=np.array([pow(int(g),j,N) for j in range(d)]); X=(np.outer(k,z)%N)/N
        wd=-(4/3)**d+np.mean(np.prod(1.5-X*(1-X),axis=1))
        if best is None or wd<best[0]: best=(wd,X)
    return best[1]
for d in [2,4,8,16]:
    for m in range(6,13):
        X=korobov(2**m,d,seed=m); out["lattice"].append({"d":d,"m":m,"WD":qmc.discrepancy(X,method="WD")})
print("lattice done", [ (q['d'],q['m'],f"{q['WD']:.2e}") for q in out["lattice"] if q['m']==12])
# ---- (b) irreducible polynomials over GF(2)
def irreducible(f,n):
    # brute force: no factor of degree <= n/2
    def pmod(a,b):
        db=b.bit_length()-1
        while a and a.bit_length()-1>=db: a^=b<<(a.bit_length()-1-db)
        return a
    for g in range(2,1<<(n//2+1)):
        if g.bit_length()-1>=1 and pmod(f,g)==0 and g!=f: return False
    return True
def polys(n):
    P=[];NP=[]
    for f in range((1<<n)+1,1<<(n+1),2):
        if irreducible(f,n):
            (P if is_primitive_gf2([(f>>i)&1 for i in range(n+1)]) else NP).append(f)
    return P,NP
degs=[4,6,6,8,8,8,8]   # dims 2..8 (dim 1 = van der Corput)
pool={n:polys(n) for n in set(degs)}
print({n:(len(p),len(q)) for n,(p,q) in pool.items()})
def pick(kind):
    used={n:0 for n in pool}; L=[]
    for n in degs:
        L.append(pool[n][0 if kind=="prim" else 1][used[n]]); used[n]+=1
    return L
def sobol_dir(f,m,rng):
    s=f.bit_length()-1; a=[(f>>(s-1-i))&1 for i in range(1,s)]   # a_1..a_{s-1}
    mm=[int(rng.integers(0,2**(i-1)))*2+1 for i in range(1,s+1)]
    for i in range(s,m):
        new=mm[i-s]^(mm[i-s]<<s)
        for k in range(1,s): new^=(a[k-1]*mm[i-k])<<k
        mm.append(new)
    return [mm[i]<<(m-1-i) for i in range(m)]   # m-bit integers
def net(polys_,m,rng):
    V=[[1<<(m-1-i) for i in range(m)]]+[sobol_dir(f,m,rng) for f in polys_]
    k=np.arange(2**m); X=np.zeros((2**m,len(V)),dtype=np.int64)
    for j,v in enumerate(V):
        for i in range(m): X[:,j]^=((k>>i)&1)*v[i]
    return X/2**m
def tval2(P,m):
    X=(P*2**m).astype(np.int64)
    for t in range(m+1):
        if all(np.all(np.bincount((X[:,0]>>(m-a))*(2**(m-t-a))+(X[:,1]>>(m-(m-t-a))),minlength=2**(m-t))==2**t) for a in range(m-t+1)): return t
m=10
for kind in ["prim","nonprim"]:
    P=pick(kind); wd={4:[],8:[]}; tv=[]
    for r in range(30):
        rng=np.random.default_rng(1000+r); X=net(P,m,rng)
        assert all(np.array_equal(np.sort(X[:,j]),np.arange(2**m)/2**m) for j in range(X.shape[1]))
        wd[4].append(qmc.discrepancy(X[:,:4],method="WD")); wd[8].append(qmc.discrepancy(X,method="WD"))
        tv.append(np.mean([tval2(X[:,[i,j]],m) for i in range(8) for j in range(i+1,8)]))
    out["irr"][kind]={"polys":[bin(f) for f in P],"WD4":[float(np.mean(wd[4])),float(np.std(wd[4]))],
                      "WD8":[float(np.mean(wd[8])),float(np.std(wd[8]))],"mean_t2":[float(np.mean(tv)),float(np.std(tv))]}
    print(kind,out["irr"][kind])
json.dump(out,open("../data/r37.json","w"),indent=1,default=float)
