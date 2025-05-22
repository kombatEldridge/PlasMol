from pyscf import dft,solvent,tools
import numpy as np
import scipy as sp

from . import diis_routine
from . import io_utils
from . import constants

class RKS():
    def __init__(self,mol):
      self.ints_factory   = mol
      self.nbf            = mol.nao 
      self.nel            = mol.nelec #returns [nalpha, nbeta]
      self.mult           = mol.multiplicity 
      self.charge         = mol.charge 
      self.nbf            = mol.nao
      self.e_nuc          = mol.energy_nuc()
      self.reference      = "rks"
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.triplet        = False

    def compute(self,options):
      print("", end ="") # print("    Alpha electrons  : %4i"%(self.nel[0]),flush=True)
      print("", end ="") # print("    Beta  electrons  : %5i"%(self.nel[1]),flush=True)
    
      self.options = options

      self.S   = self.ints_factory.intor('int1e_ovlp') 
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')

      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("", end ="") # print("\n    Exchange-Correlation Functional:",options.xc)
      print("", end ="") # print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("", end ="") # print("\n    Hybrid alpha parameter: %f"%options.xcalpha)
      #dft grid parameters
      self.jk    = dft.RKS(self.ints_factory,options.xc) #jk object from pyscf
      #self.jk.multiplicity=2

      eps_scaling = np.zeros((self.nbf,self.nbf))
      self.eps = np.zeros((self.nbf))

      # if self.options.relativistic == "zora":
      #   zora = relativistic.ZORA(self)
      #   zora.get_zora_correction()
      #   self.T =  zora.T[0]
      #   self.H_so = zora.H_so 
      #   eps_scaling = zora.eps_scal_ao
      #   if self.options.so_scf is True:
      #     exit("    CAN'T ADD H_SO TO RKS")

      if options.guess_mos_provided is True:
        print("", end ="") # print("    Reading MOs from File...")
        Cmo, nel = io_utils.read_real_mos(options.guess_mos)
        #sanity check
        if Cmo.shape != (2,self.nbf,self.nbf):
          exit("Incompatible MOs dimension")
        elif nel[0] != nel[1]:
          exit("Incompatible number of electrons, Na = %i, Nb = %i"%(nel[0],nel[1]))
        D = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))
      else:
        self.jk.conv_tol	= options.e_conv
        self.jk.conv_tol_grad = options.d_conv
        self.jk.verbose = 4
        energy_pyscf = self.jk.kernel()
        Cmo = np.zeros((self.nbf,self.nbf))
        print("", end ="") # print("    Computing MOs from PySCF....")
        #D =  self.jk.init_guess_by_atom()
        C =  self.jk.mo_coeff
        D = C[:,:self.nel[0]]@C.T[:self.nel[0],:]

      F = self.T + self.Vne

      #orthogonalization matrix
      Shalf = sp.linalg.sqrtm(np.linalg.inv(self.S))
      Forth = np.matmul(Shalf,np.matmul(F,Shalf))
      newE  = np.einsum("mn,mn->",D,(F+F)) #+ Exc 
      newD  = np.zeros((self.nbf,self.nbf))

      if self.options.cosmo is True: 
        cosmo = solvent.ddcosmo.DDCOSMO(self.ints_factory)
        cosmo.eps = self.options.cosmo_epsilon 
        print("", end ="") # print("    COSMO solvent enabled. Dielectric constant %f"%(self.options.cosmo_epsilon))

      energy = 0.
      print("", end ="") # print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ",flush=True)
      if options.diis is True:
        diis = diis_routine.DIIS(self)
        err_vec = np.zeros((1,1,1))
        Fdiis = np.zeros((1,1,1))
  
      for it in range(options.maxiter):
          ##THIS WORKS
          pot = self.jk.get_veff(self.ints_factory,2.*D) #pyscf effective potential
          if self.options.cosmo is True:
            e_cosmo, v_cosmo = cosmo.kernel(dm=2.*D)
            F = self.T + self.Vne + pot + v_cosmo #2.*J - 0*K + Vxc
          else:
            F = self.T + self.Vne + pot  #2.*J - 0*K + Vxc
          ###
          if options.diis is True:
            Faorth, e_rms = diis.do_diis(F,D,self.S,Shalf)
          else:
            diis = False
            Faorth = np.matmul(Shalf.T,np.matmul(F,Shalf))
          evals, evecs = sp.linalg.eigh(Faorth)
          C    = np.matmul(Shalf,evecs).real
          if options.noscf is True:
            print("", end ="") # print("    NOSCF flag enabled. Guess orbitals will be used without optimization")
            dE = options.e_conv/10.
            dD = options.d_conv/10.
            oneE  = 2.*np.trace(np.matmul(D,self.T+self.Vne)) 
            if self.options.cosmo is True:
              oneE  += e_cosmo 
            twoE  = pot.ecoul + pot.exc 
            newE  = oneE + twoE
            energy = newE
          else:
            newD = np.matmul(C[:self.nbf,:self.nel[0]],C.T[:self.nel[0],:self.nbf])

            if self.options.cosmo is True:
              oneE = 2. * np.trace(np.matmul((newD),(self.T+self.Vne))) + e_cosmo
            else:
              oneE = 2. * np.trace(np.matmul((newD),(self.T+self.Vne)))
            twoE = pot.ecoul + pot.exc #1. * np.trace(np.matmul(newD,(2.*J - 0.*K))) + Exc
            newE = oneE + twoE
            dE   = abs(newE - energy)
            if options.diis is True:
                dD = e_rms
            else:
                dD = abs(np.sqrt(np.trace((newD-D)**2)))
            energy = newE
            D = 1.*newD
            if it > 0:
                print("", end ="") # print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,diis.is_diis),flush=True)

          if (dE < options.e_conv) and (dD < options.d_conv):
            # if self.options.relativistic == "zora":
            #   Ci = np.zeros((1,self.nbf),dtype=complex)
            #   zora_scal = 0.
            #   for i in range(self.nel[0]):
            #     Ci[0] = C.T[i]
            #     Di = np.matmul(np.conjugate(Ci.T),Ci).real
            #     eps_scal = np.trace(np.matmul(eps_scaling,Di))
            #     self.eps[i] = evals[i]/(1.+eps_scal)
            #     zora_scal -= eps_scal*self.eps[i]
             
            #   self.eps[self.nel[0]:] = evals[self.nel[0]:]
            # else:
            self.eps = evals  

            print("", end ="") # print("    Iterations have converged!")
            print("", end ="") # print("")
            print("", end ="") # print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
            print("", end ="") # print("    One-electron Energy:          %20.12f" %(oneE))
            print("", end ="") # print("    Two-electron Energy:          %20.12f" %(twoE))
            print("", end ="") # print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
            print("", end ="") # print("")
            print("", end ="") # print("    Orbital Energies [Eh]")
            print("", end ="") # print("    ---------------------")
            print("", end ="") # print("")
            print("", end ="") # print("    Alpha occupied:")
            for i in range(self.nel[0]):
                print("", end ="") # print("    %4i: %12.5f "%(i+1,self.eps[i]),end="")
                if (i+1)%3 == 0: print("", end ="") # print("")
            print("", end ="") # print("")
            print("", end ="") # print("    Alpha Virtual:")
            for a in range(self.nbf-self.nel[0]):
                print("", end ="") # print("    %4i: %12.5f "%(self.nel[0]+a+1,self.eps[self.nel[0]+a]),end="")
                if (a+1)%3 == 0: print("", end ="") # print("")
            print("", end ="") # print("")
            break
          if (it == options.maxiter-1):
              print("", end ="") # print("SCF iterations did not converge!")
              exit(0)
          it += 1
      print("", end ="") # print("")

      print("", end ="") # print("    Molecular Orbital Analysis")
      print("", end ="") # print("    --------------------------\n")
  
      ao_labels = self.ints_factory.ao_labels()
      for p in range(self.nbf):
        print("", end ="") # print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (2 if p < self.nel[0] else 0),self.eps[p]))
        print("", end ="") # print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(C.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("", end ="") # print("    %-12s: % 8.5f "%(ao_labels[i],C[i][p]),end="")
          if ((idx+1)%3 == 0): print("", end ="") # print("")
        print("", end ="") # print("\n")

      self.F = [F,F]
      self.C = [C,C]
      self.D = [D,D]

      self.eps = [self.eps,self.eps]
      self.scf_energy = twoE + oneE + self.e_nuc
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([C,C],self.nel,mos_filename)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      print("", end ="") # print("\n")
      print("", end ="") # print("    Mulliken Population Analysis (q_A = Z_A - Q_A)")
      print("", end ="") # print("    --------------------------------------------\n")
      Q = 2.*np.diag(D@self.S)
      natoms = int(ao_labels[-1].split()[0])+1
      qA = np.zeros(natoms)
      QA = np.zeros(natoms)
      ZA = np.zeros(natoms)
      A = [None]*natoms
      
      for p in range(self.nbf):
          atom_index = ao_labels[p].split()[0]
          atom_label = ao_labels[p].split()[1]
          ZA[int(atom_index)]  = constants.Z[atom_label.upper()]
          QA[int(atom_index)] += Q[p] 
          A[int(atom_index)] = atom_index + " " + atom_label
      for a in range(natoms):
          print("", end ="") # print("    %-12s: % 8.5f "%(A[a],ZA[a]-QA[a]))
      print("", end ="") # print("\n")
      print("", end ="") # print("\n    Energetics Summary")
      print("", end ="") # print("    ------------------\n")
      print("", end ="") # print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
      print("", end ="") # print("    One-electron Energy:          %20.12f" %(oneE))
      print("", end ="") # print("    Two-electron Energy:          %20.12f" %(twoE))
      print("", end ="") # print("    @RKS Final Energy:            %20.12f" %(self.scf_energy))

      #quick spin analysis 
      N = np.trace((D@self.S))
      self.S2 = N - np.trace((D@self.S)@(D@self.S))
      print("", end ="") # print("")
      print("", end ="") # print("    Computed <S2> : %12.8f "%(self.S2))
      print("", end ="") # print("    Expected <S2> : %12.8f "%(0.0))
           

      return self.scf_energy
