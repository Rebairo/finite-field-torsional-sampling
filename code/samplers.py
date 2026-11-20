import numpy as np
from scipy.stats import qmc
PRIM={6:0b1000011,7:0b10000011,8:0b100011101,9:0b1000010001,10:0b10000001001,11:0b100000000101,
      12:0b1000001010011,13:0b10000000011011,14:0b100000000101011,15:0b1000000000000011,16:0b10000000000101101}
def mseq(m):
    f=PRIM[m]; s=1; S=np.empty(2**m-1,dtype=np.int64)
    for k in range(2**m-1):
        S[k]=s; s<<=1
        if s>>m&1: s^=f
    return S
def sym_shift(m,d):
    """one primitive polynomial; dimension j = cyclic shift of the same m-sequence (shared Frobenius phase)"""
    S=mseq(m); P=len(S); lag=P//d
    X=np.stack([np.roll(S,-j*lag) for j in range(d)],1)
    return np.vstack([np.zeros((1,d),dtype=np.int64),X])/2**m
def sym_frob(m,d):
    """dimension j = Frobenius conjugate alpha^{k 2^j}; exactly Galois-symmetric (needs d<=m)"""
    S=mseq(m); P=len(S); k=np.arange(P)
    X=np.stack([S[(k*pow(2,j,P))%P] for j in range(d)],1)
    return np.vstack([np.zeros((1,d),dtype=np.int64),X])/2**m
def sobol(m,d,seed=None):
    return qmc.Sobol(d=d,scramble=seed is not None,seed=seed).random_base2(m)
def rand(m,d,seed): return np.random.default_rng(seed).random((2**m,d))
