"""Bouchaud trap landscape on the torus (k_B T = 1).
Each torsion carries M narrow Gaussian wells on a flat top; the well depths are the stratified quantiles
eps_j = -T_g ln((j-1/2)/M) of an exponential distribution with mean T_g = 1/gamma, randomly placed.
Escape times ~ exp(eps) are then Pareto distributed with tail exponent gamma (Bouchaud 1992).
Only the nearest well contributes (wells are separated by > 8 widths), so u is exact to ~1e-14."""
import numpy as np
TWO_PI=2*np.pi
class Trap:
    def __init__(self,gamma,M=128,sigma=0.006,seed=0):
        rng=np.random.default_rng(seed); Tg=1.0/gamma
        self.eps=rng.permutation(-Tg*np.log((np.arange(M)+0.5)/M)); self.M=M; self.kappa=1/sigma**2; self.gamma=gamma
        self.h=TWO_PI/M
    def u(self,t):
        t=np.mod(t,TWO_PI); j=np.minimum((t/self.h).astype(int),self.M-1); c=(j+0.5)*self.h
        return -self.eps[j]*np.exp(self.kappa*(np.cos(t-c)-1))
    def exact1(self,K=2**18):
        g=(np.arange(K)+0.5)*TWO_PI/K; u=self.u(g); w=np.exp(-u); z=w.mean()
        return np.log(z),(u*w).sum()/w.sum(),g,w/w.sum()
