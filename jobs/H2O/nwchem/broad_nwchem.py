#!/opt/ohpc/pub/apps/uofm/python/3.12.1/bin/python

import numpy as np
import sys

warray = sys.argv[1].split(",") #tuple
dump   = float(sys.argv[2]) #float
f_in   = open(sys.argv[3],"r") #file

roots = []
os    = []

for line in f_in:
#    if "Root" in line.split() and "singlet" in line.split():
    if "Root   1 triplet" in line:
        break
    if "Root" in line.split():
        if (line.split()[5] == "a.u."):
            roots.append(float(line.split()[6]))
        else:
            roots.append(float(line.split()[5]))
    if "Total Oscillator Strength" in line:
        os.append(float(line.split()[3]))
#    if "Dipole Oscillator Strength" in line:
#        os.append(float(line.split()[3]))
#    if "Transition Moments    X" in line:
#        os.append([float(line.split()[3]),float(line.split()[5]),float(line.split()[7])])

#for i in range(len(os)):
#    print("%5.6f %5.16f %5.16f %5.16f" %(roots[i],os[i][0],os[i][1],os[i][2]))
#exit(0)

wrange = np.zeros(3)
if len(warray) < 3:
  wrange[0] = roots[0] - 5.    
  wrange[1] = roots[len(roots)-1] + 5.    
  wrange[2] = float(warray[0])
else:
  wrange[0] = float(warray[0])    
  wrange[1] = float(warray[1])     
  wrange[2] = float(warray[2])
 
nw = int((float(wrange[1]) - float(wrange[0]))/float(wrange[2]))
for windex in range(nw):
    w = (float(wrange[0]) + float(wrange[2]) * windex)
    S = 0. 
    for root in range(len(roots)):
        S += os[root]*dump/((w-roots[root])**2 + dump**2)
    S /= (1.0 * np.pi)
    print("%5.6f %5.16f" %(w,S))





