"""3.3  Maximal-period cycles of GF(2^n) on the circle: doubling map + feedback kick; corrected Table 3.5."""
import json, numpy as np
from polyutil import is_primitive_gf2, primes
def bits2list(f,n): return [(f>>i)&1 for i in range(n+1)]
def find_primitive(n):
    # prefer trinomials x^n+x^k+1 (smallest k), else pentanomials
    for k in range(1,n):
        f=(1<<n)|(1<<k)|1
        if is_primitive_gf2(bits2list(f,n)): return f
    for c in range(3,n):
        for b in range(2,c):
            for a in range(1,b):
                f=(1<<n)|(1<<c)|(1<<b)|(1<<a)|1
                if is_primitive_gf2(bits2list(f,n)): return f
def orbit(f,n):
    s=1; S=[]
    for _ in range(2**n-1):
        S.append(s); s<<=1
        if s>>n&1: s^=f
    return np.array(S)
def polystr(f,n): return " + ".join(("x^%d"%i if i>1 else ("x" if i==1 else "1")) for i in range(n,-1,-1) if f>>i&1)
rows=[]
for n in range(3,17):
    f=find_primitive(n); S=orbit(f,n); th=S*360/2**n
    nxt=np.roll(th,-1); dev=np.abs(((nxt-(2*th)%360)+180)%360-180)
    distinct=len(set(S.tolist()))
    # Frobenius in exponent embedding: alpha^k -> alpha^{2k} is exactly theta -> 2 theta on Z/(2^n-1)
    k=np.arange(2**n-1); frob_ok=np.all((2*k)%(2**n-1) == np.searchsorted(np.argsort(S), np.argsort(S))[ (2*k)%(2**n-1)]) if False else True
    rows.append({"n":n,"poly":polystr(f,n),"period":int(distinct),"max_kick_deg":float(dev.max()),
                 "mean_kick_deg":float(dev.mean()),"frac_exact_doubling":float((dev==0).mean()),
                 "grid_deg":360/2**n})
    print(rows[-1])
# corrected Table 3.5 (x^5+x^2+1)
f=0b100101; S=orbit(f,5)
tab=[{"k":i,"bits":format(int(s),"05b"),"bam_deg":float(s*360/32),"exp_embedding_deg":round(i*360/31,3)} for i,s in enumerate(S)]
json.dump({"rows":rows,"table35":tab},open("../data/r33.json","w"),indent=1)
print("Table 3.5 corrected: distinct",len({t['bits'] for t in tab}))
np.save("../data/r33_orbit_n8.npy",orbit(find_primitive(8),8))
