import numpy as np
rng0=np.random.default_rng(7); K=4
amp=rng0.normal(size=K)/np.arange(1,K+1); ph=rng0.uniform(0,2*np.pi,K)
_g=np.linspace(0,2*np.pi,4096,endpoint=False)
_s=sum(amp[k]*np.cos((k+1)*_g+ph[k]) for k in range(K)).std()
C=0.5
def make(name):
    if name.startswith("funnel"):
        A=float(name.split("_")[1]); u=lambda t:-A*np.cos(t); du=lambda t:A*np.sin(t)
    else:
        e=float(name.split("_")[1])
        u=lambda t:e*sum(amp[k]*np.cos((k+1)*t+ph[k]) for k in range(K))/_s
        du=lambda t:-e*sum(amp[k]*(k+1)*np.sin((k+1)*t+ph[k]) for k in range(K))/_s
    return u,du
def U(th,u):  # th radians, shape (...,d)
    return u(th).sum(-1)+C*np.cos(th[...,:-1]-th[...,1:]).sum(-1)
def gradU(th,du):
    g=du(th); s=np.sin(th[...,:-1]-th[...,1:])
    g[...,:-1]+=-C*s; g[...,1:]+=C*s; return g
def exact(u,d,beta=1.0,M=1024):
    g=np.linspace(0,2*np.pi,M,endpoint=False); w=1.0/M
    a=np.exp(-beta*u(g)); T=np.exp(-beta*C*np.cos(g[:,None]-g[None,:]))
    v=a*w; lz=0.0
    for _ in range(d-1):
        v=(v@T)*a*w; s=v.sum(); lz+=np.log(s); v/=s
    return lz+np.log(v.sum())
def exact_EU(u,d):
    h=1e-4; return -(exact(u,d,1+h)-exact(u,d,1-h))/(2*h)
def exact_marginal(u,d,i,M=1440,nb=36):
    g=np.linspace(0,2*np.pi,M,endpoint=False); a=np.exp(-u(g)); T=np.exp(-C*np.cos(g[:,None]-g[None,:]))
    al=[a/a.sum()]
    for _ in range(d-1): v=(al[-1]@T)*a; al.append(v/v.sum())
    be=[np.ones(M)]
    for _ in range(d-1): v=T@(a*be[-1]); be.append(v/v.sum())
    be=be[::-1]; p=al[i]*be[i]; p/=p.sum()
    return p.reshape(nb,-1).sum(1)
