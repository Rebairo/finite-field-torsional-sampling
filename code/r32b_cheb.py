import json, numpy as np
from fractions import Fraction as Fr
from polyutil import *
from math import factorial
P=primes(6000)
summary=[]
for g in [Fr(1,3),Fr(2,3),Fr(1,5),Fr(3,5)]:
    for L in [4,6,8,10,12]:
        c=charpoly(g,Fr(1),L)
        good=0; full=0; first=None; bad2 = cycle_type(c,2) is None
        for p in P:
            ct=cycle_type(c,p)
            if ct is None: continue
            good+=1
            if ct==(L,):
                full+=1; first=first or p
        summary.append({"gamma":str(g),"L":L,"bad_at_2":bad2,"first_prime_Lcycle":first,
                        "freq_Lcycle":full/good,"S_L_expected":1/L,"good_primes":good,
                        "lead_coeff_2adic":(lambda x:(x & -x).bit_length()-1)(abs(c[-1]))})
        print(summary[-1])
json.dump(summary,open("../data/r32b_cheb.json","w"),indent=1)
