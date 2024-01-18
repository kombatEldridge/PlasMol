import numpy as np
import os
from pyscf import dft
import time
import utils
import constants

class ZORA():
   
    def __init__(self,wfn):
      self.ints = wfn.ints_factory 
      self.wfn  = wfn
      self.options = wfn.options
   

    def _read_basis(self,atoms,comp):
       f = open(os.path.abspath(os.path.dirname(__file__)+"/basis/modbas.")+str(comp)+"c","r")
     
       self.alphas = []
       self.coeffs = []
       self.nbasis = np.zeros((len(atoms)))
       for atom, a in enumerate(atoms):
         N = 0
         count  = 0
         for line in f:
           if line.split()[0] == a.lower():
             N = int(line.split()[2])
             self.nbasis[atom] = N
             count  = -1
           
           if count > 0 and count < N+1:
             self.alphas.append(float(line.split()[0]))
             self.coeffs.append(float(line.split()[1]))
    
           count += 1
         f.seek(0) 
    
    
    def compute_zora_kernel(self):
        #get grid
        molecule = self.wfn.ints_factory
        self.wfn.jk.grids.level = self.options.grid_level
        
        atomic_grid = self.wfn.jk.grids.gen_atomic_grids(molecule)
       
        # get zora potential
        self._read_basis(self.options.molecule["atoms"],2)
        self.nbasis = np.asarray(self.nbasis,dtype=int)
        centers = molecule.atom.split(";") #+str(Ci+1)]
    
        self.kernel  = []
        self.points  = []
        for Ci, C in enumerate(self.options.molecule["atoms"]):
          coords_C = np.asarray(centers[Ci].split()[1:], dtype=float)
          dim = len(atomic_grid[C][1])
          pts = atomic_grid[C][0] + coords_C 
          val  = np.zeros((dim))
          offset = 0
          for Ai, A in enumerate(self.options.molecule["atoms"]):
            coords_A = np.asarray(centers[Ai].split()[1:], dtype=float)
            PA = pts - coords_A 
            RPA = np.sum(PA**2, axis=1)
            c = self.coeffs[offset:offset+self.nbasis[Ai]]                                             
            a = self.alphas[offset:offset+self.nbasis[Ai]]
            boys = utils.compute_mBoys(0,np.outer(a,RPA))
            val += 2.0 * np.einsum("i,i,ip->p",c,np.sqrt(a),boys,optimize=True)/np.sqrt(np.pi) 
            offset += self.nbasis[Ai]
            val -= constants.Z[A.upper()]/np.sqrt(RPA)
          self.kernel += list(val)
          self.points  += list(pts)
    
        _, self.weights = self.wfn.jk.grids.get_partition(molecule,atomic_grid)
        self.kernel  = np.asarray(self.kernel)
        self.points  = np.asarray(self.points)
        self.weights = np.asarray(self.weights)
        
        print("   ZORA grid computed successfuly!",flush=True)
    

    def get_zora_correction(self):
        print("    Computing ZORA integrals integrals.",flush=True)
        tic = time.time()
        self.compute_zora_kernel()
        nbf = self.wfn.nbf
        self.eps_scal_ao = np.zeros((nbf,nbf))
        self.T = np.zeros((4,nbf,nbf))

        npoints = len(self.points)
        print("    Number of grid points: %i"%npoints)
        batch_size = self.options.batch_size
        excess = npoints%batch_size
        nbatches = (npoints-excess)//batch_size
        print("    Number of batches: %i"%(nbatches+1))
        print("    Maximum Batch Size: %i"%batch_size)
        print("    Memory estimation for ZORA build: %8.4f mb"%(batch_size*nbf*6*8/1024./1024.),flush=True)
        for batch in range(nbatches+1):
          low = batch*batch_size
          if batch < nbatches:
            high = low+batch_size
          else:
            high = low+excess

          bpoints  = self.points[low:high]
          bweights = self.weights[low:high]
          bVzora   = self.kernel[low:high]
          ao_val = dft.numint.eval_ao(self.ints, bpoints, deriv=1)
          kernel = 1./(2.*(137.036**2) - bVzora)
          self.T[0] += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel,optimize=True) * (137.036**2)
          self.eps_scal_ao += np.einsum("xip,xiq,i->pq",ao_val[1:],ao_val[1:],bweights*kernel**2,optimize=True) * (137.036**2)
          kernel = bVzora/(4.*(137.036**2) - 2.*bVzora)
          self.T[1] += np.einsum("ip,iq,i->pq",ao_val[2],ao_val[3],bweights*kernel,optimize=True)
          self.T[1] -= np.einsum("ip,iq,i->pq",ao_val[3],ao_val[2],bweights*kernel,optimize=True)

          self.T[2] += np.einsum("ip,iq,i->pq",ao_val[3],ao_val[1],bweights*kernel,optimize=True)
          self.T[2] -= np.einsum("ip,iq,i->pq",ao_val[1],ao_val[3],bweights*kernel,optimize=True)

          self.T[3] += np.einsum("ip,iq,i->pq",ao_val[1],ao_val[2],bweights*kernel,optimize=True)
          self.T[3] -= np.einsum("ip,iq,i->pq",ao_val[2],ao_val[1],bweights*kernel,optimize=True)
        toc = time.time()

        print("    ZORA integrals computed in %5.2f seconds \n"%(toc-tic),flush=True)

        self.H_so = np.zeros((2*nbf,2*nbf),dtype=complex)
        Kx = 1j * self.T[1]
        Ky = 1j * self.T[2]
        Kz = 1j * self.T[3]

        self.H_so[:nbf,:nbf] =   Kz
        self.H_so[nbf:,nbf:] =  -Kz
        self.H_so[:nbf,nbf:] =  (Kx - 1j*Ky)
        self.H_so[nbf:,:nbf] =  (Kx + 1j*Ky)
        