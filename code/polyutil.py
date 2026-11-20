from fractions import Fraction as Fr
from math import lcm, gcd
from functools import reduce
def glw(g,L,exact=False):
    w=[Fr(1) if exact else 1.0]
    for j in range(1,L+1): w.append(w[-1]*((1-(Fr(g)+1)/j) if exact else (1-(g+1)/j)))
    return w

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

