"""Model Ramachandran distribution: three bivariate Gaussian basins (alpha-helix, beta-sheet, left-handed helix),
each normalised to unit peak, summed. Evaluated periodically (images at +/-360 deg)."""
import numpy as np
BASINS={"alpha":((-60,-45),[[1000,200],[200,500]]),"beta":((-120,120),[[1200,300],[300,900]]),"alphaL":((60,60),[[800,100],[100,700]])}
def density(phi,psi):
    tot=0
    for mu,S in BASINS.values():
        Si=np.linalg.inv(np.array(S,float)); comp=0
        for a in (-360,0,360):
            for b in (-360,0,360):
                x=phi-mu[0]+a; y=psi-mu[1]+b
                comp=comp+np.exp(-0.5*(Si[0,0]*x*x+2*Si[0,1]*x*y+Si[1,1]*y*y))
        tot=tot+comp          # each basin normalised to unit peak
    return tot
