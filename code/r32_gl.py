"""3.2  Fractional relaxation D^g y = -lam y as a Grünwald–Letnikov recurrence.
Memory truncated at L steps -> order-L linear recurrence -> characteristic polynomial with RATIONAL
coefficients when g and mu=lam*h^g are rational. Reduce mod primes: Dedekind cycle types; mod 2 primitivity."""
import numpy as np, json
from fractions import Fraction as Fr
from math import lcm, gcd
from functools import reduce
from scipy.special import erfcx

def glw(g,L,exact=False):
    w=[Fr(1) if exact else 1.0]
    for j in range(1,L+1): w.append(w[-1]*((1-(Fr(g)+1)/j) if exact else (1-(g+1)/j)))
    return w

# ---- (a) numerical check: GL scheme reproduces Mittag-Leffler; memory truncation error
def gl_solve(g,lam,h,nsteps,L=None):
    w=np.array(glw(g,nsteps)); y=np.zeros(nsteps+1); y[0]=1.0; mu=lam*h**g
    for n in range(1,nsteps+1):
        m=n if L is None else min(n,L)
        hist=np.dot(w[1:m+1],y[n-1::-1][:m])
        y[n]=(-hist + y[0]*w[:m+1].sum())/(1+mu)
    return y
g=0.5; h=0.01; T=50; ns=int(T/h); tt=np.arange(ns+1)*h
ex=erfcx(np.sqrt(tt))
full=gl_solve(g,1,h,ns)
res={"full_memory_max_abs_err":float(np.max(abs(full-ex)))}
print("full memory GL vs ML max abs err",res["full_memory_max_abs_err"])
trunc={}
for L in [10,30,100,300,1000,3000]:
    y=gl_solve(g,1,h,ns,L); trunc[L]=float(np.max(abs(y-ex)))
    print("L",L,"max abs err",trunc[L])
res["truncation"]=trunc
np.savez("../data/r32_trunc.npz",t=tt,exact=ex,full=full,**{f"L{L}":gl_solve(g,1,h,ns,L) for L in [10,100,1000]})

# ---- (b) exact characteristic polynomials and reductions
def charpoly(g,mu,L):
    w=glw(g,L,True)
    # recurrence (1+mu) y_n + sum_{j=1}^L w_j y_{n-j} = forcing
    # P(z) = (1+mu) z^L + sum_j w_j z^{L-j};  coefficients low->high
    c=[None]*(L+1); c[L]=1+mu
    for j in range(1,L+1): c[L-j]=w[j]
    D=reduce(lcm,[x.denominator for x in c]); ci=[int(x*D) for x in c]
    G=reduce(gcd,ci); return [x//G for x in ci]

# GF(p) polynomial arithmetic, lists low->high
def trim(a):
    while a and a[-1]==0: a.pop()
    return a
def pmod(a,b,p):
    a=trim([x%p for x in a]); b=trim([x%p for x in b]); inv=pow(b[-1],p-2,p)
    while len(a)>=len(b):
        c=a[-1]*inv%p; s=len(a)-len(b)
        for i,bi in enumerate(b): a[s+i]=(a[s+i]-c*bi)%p
        trim(a)
    return a
def pmul(a,b,p):
    r=[0]*(len(a)+len(b)-1) if a and b else []
    for i,x in enumerate(a):
        if x:
            for j,y in enumerate(b): r[i+j]=(r[i+j]+x*y)%p
    return trim(r)
def pgcd(a,b,p):
    a=trim([x%p for x in a]); b=trim([x%p for x in b])
    while b: a,b=b,pmod(a,b,p)
    if a: inv=pow(a[-1],p-2,p); a=[x*inv%p for x in a]
    return a
def pdiv(a,b,p):
    a=trim([x%p for x in a]); inv=pow(b[-1],p-2,p); q=[0]*max(len(a)-len(b)+1,1)
    while len(a)>=len(b):
        c=a[-1]*inv%p; s=len(a)-len(b); q[s]=c
        for i,bi in enumerate(b): a[s+i]=(a[s+i]-c*bi)%p
        trim(a)
    return trim(q)
def ppow(base,e,f,p):
    r=[1]; b=pmod(base,f,p)
    while e:
        if e&1: r=pmod(pmul(r,b,p),f,p)
        b=pmod(pmul(b,b,p),f,p); e>>=1
    return r
def deriv(a,p): return trim([(i*a[i])%p for i in range(1,len(a))])
def cycle_type(f,p):
    """None if bad reduction (leading coeff divisible by p or not squarefree)."""
    f=[x%p for x in f]
    if f[-1]==0: return None
    f=trim(f)
    if len(pgcd(f,deriv(f,p),p))>1: return None
    degs=[]; h=[0,1]; i=0
    while len(f)>1:
        i+=1; h=ppow(h,p,f,p)
        d=pgcd(f,trim([(x-(1 if k==1 else 0))%p for k,x in enumerate(h+[0]*(2-len(h)))]),p)
        if len(d)>1:
            degs+= [i]*((len(d)-1)//i); f=pdiv(f,d,p); h=pmod(h,f,p) if len(f)>1 else h
        if 2*i>len(f)-1 and len(f)>1: degs.append(len(f)-1); break
    return tuple(sorted(degs,reverse=True))
def primes(n): return [q for q in range(2,n) if all(q%r for r in range(2,int(q**.5)+1))]
def is_primitive_gf2(f):
    d=len(f)-1; M=2**d-1
    if ppow([0,1],M,f,2)!=[1]: return False
    return all(ppow([0,1],M//q,f,2)!=[1] for q in primes(M+1) if M%q==0)
def galois_verdict(f,L,P):
    types={}
    for p in P:
        ct=cycle_type(f,p)
        if ct: types.setdefault(ct,p)
    has=lambda pred: any(pred(t) for t in types)
    full_cycle=has(lambda t:t==(L,))
    nm1=has(lambda t:sorted(t)==[1,L-1])
    transp=has(lambda t: sum(1 for x in t if x%2==0)==1 and any(x==2 for x in t) and all(x%2==1 or x==2 for x in t))
    # Jordan-type criterion: transitive (L-cycle) + (L-1)-cycle + an element whose power is a transposition => S_L
    return types, (full_cycle and nm1 and transp)

rows=[]
P=primes(400)
for g in [Fr(1,3),Fr(2,3),Fr(1,5),Fr(3,5)]:
    for mu in [Fr(1),Fr(1,3)]:
        for L in [3,4,5,6,8,10,12,16]:
            c=charpoly(g,mu,L); rec=c[::-1]          # reciprocal polynomial: same splitting field
            out={"gamma":str(g),"mu":str(mu),"L":L}
            for name,f in [("P",c),("Prec",rec)]:
                ct=cycle_type(f,2); out[name+"_mod2"]=ct
                if ct==(L,): out[name+"_primitive_mod2"]=is_primitive_gf2(trim([x%2 for x in f]))
            types,SL=galois_verdict(c,L,P); out["S_L"]=SL; out["n_cycle_types"]=len(types)
            rows.append(out)
            print(out)
res["rows"]=rows
# example from the text: x^3-3x-1
ex3=[-1,-3,0,1]; res["x3-3x-1"]={"mod2":cycle_type(ex3,2),"primitive":is_primitive_gf2([1,1,0,1])}
print(res["x3-3x-1"])
json.dump(res,open("../data/r32.json","w"),indent=1,default=str)
