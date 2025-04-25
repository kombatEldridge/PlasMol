import numpy as np
import scipy as sp
import os
import sys
from scipy import special

#get basis set information
def read_basis(basis, atoms):
    d = {}
    a = {}
    l = {}
    f = open(sys.path.insert(1,os.path.abspath(os.path.dirname(__file__)+"/basis/"))+basis["name"]+".gbs","r")
    count = 0
    shell = 0
    atype = None
    nprim = 1e5
    alpha  = {}
    contr  = {}
    amomns = []
    for line in f:
        if count == (nprim + 3):
            count = 2

        if count > 2:
            line = line.replace("D","e",3)
            for i, subshell in enumerate(amomn):
                if count == 3:
                    alpha[str(shell)+"-"+subshell] = []
                    contr[str(shell)+"-"+subshell] = []
                alpha[str(shell)+"-"+subshell].append(float(line.split()[0]))
                contr[str(shell)+"-"+subshell].append(float(line.split()[i+1]))
            count += 1

        if count == 2:
            if ("****" in line):
                a[atype] = alpha
                d[atype] = contr
                l[atype] = amomns
                count = 0
            else:
                shell += 1 
                amomn = line.split()[0]
                for subshell in amomn:
                    amomns.append(str(shell)+"-"+subshell)
                nprim = int(line.split()[1])   
                count += 1
        
        if count == 1:
            if line.split()[0] in atoms:
                alpha  = {}
                contr  = {}
                amomns = []
                atype = line.split()[0]
                shell = 0
                count += 1

        if ("****" in line) and (count == 0):
            count += 1

    return l, d, a       

def compute_Boys(index,x):
    if x < 1e-7:
        return (2*index+1)**(-1) - x*(2*index+3)**(-1)
    else:
        return 0.5 * x**(-(index+0.5)) * sp.special.gamma(index+0.5) * sp.special.gammainc(index+0.5,x)

def compute_mBoys(index,x):
    return 0.5 * x**(-(index+0.5)) * sp.special.gamma(index+0.5) * sp.special.gammainc(index+0.5,x)

def lorentzian(wrange,dump,roots,os):
    nw = int((float(wrange[1]) - float(wrange[0]))/float(wrange[2]))
    w = np.zeros(nw)
    S = np.zeros(nw)
    for windex in range(nw):
        w[windex] = (float(wrange[0]) + float(wrange[2]) * windex)
        for root in range(len(roots)):
            S[windex] += os[root]*dump/((w[windex]-roots[root])**2 + dump**2)
        S[windex] /= (1.0 * np.pi)
    return w, S    
