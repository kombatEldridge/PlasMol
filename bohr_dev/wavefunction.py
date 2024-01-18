from pyscf import scf,dft,solvent,tools
import numpy as np
import scipy as sp
import diis_routine
import time
import relativistic
import io_utils
import constants

class RHF():

    def __init__(self,mol):
      
      self.ints_factory = mol
      self.nbf          = mol.nao 
      self.nel          = mol.nelec #returns [nalpha, nbeta]
      self.mult         = mol.multiplicity 
      self.charge       = mol.charge 
      self.nbf		= mol.nao
      self.e_nuc	= mol.energy_nuc()
      self.reference    = "rhf"
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.triplet      = False

    def compute(self,options):
      print("    Alpha electrons  : %4i"%(self.nel[0]))
      print("    Beta  electrons  : %5i"%(self.nel[1]))

      self.options = options
      self.S   = self.ints_factory.intor('int1e_ovlp') 
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')
   
      self.frequencies = options.frequencies

      self.jk = scf.RHF(self.ints_factory) #jk object from pyscf
      self.grids_helper = scf.RKS(self.ints_factory) #jk object from pyscf
      self.jk.grids = self.grids_helper.grids
      eps_scaling = np.zeros((self.nbf,self.nbf))
      self.eps = np.zeros((self.nbf))

      if self.options.relativistic == "zora":
        zora = relativistic.ZORA(self)
        zora.get_zora_correction()
        self.T =  zora.T[0]
        self.H_so = zora.H_so
        eps_scaling = zora.eps_scal_ao
        if self.options.so_scf is True:
          exit("    CAN'T ADD H_SO TO UHF")
  
      F = self.T + self.Vne

      #orthogonalization matrix
      Shalf = sp.linalg.sqrtm(np.linalg.inv(self.S))
      Forth = np.matmul(Shalf,np.matmul(F,Shalf))

      #find trial movecs
      evals, evecs = np.linalg.eigh(Forth)
      D            = self.jk.init_guess_by_atom()
      
      newE  = np.einsum("mn,mn->",D,(F+F)) 
      newD  = np.zeros((self.nbf,self.nbf))

      energy = 0.
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ",flush=True)
      if options.diis is True:
        err_vec = np.zeros((1,1,1))
        Fdiis = np.zeros((1,1,1))
  
      for it in range(options.maxiter):
          J, K = self.jk.get_jk(dm=D)
          
          F = self.T + self.Vne + 2.*J - K
          if options.diis is True:
            Faorth, err_vec, e_rms, Fdiis, is_diis = diis_routine.do_diis(F,D,self.S,Shalf,np.asarray(err_vec),Fdiis)
          else:
            diis = False
            Faorth = np.matmul(Shalf.T,np.matmul(F,Shalf))
          evals, evecs = np.linalg.eigh(Faorth)
          C    = np.matmul(Shalf,evecs).real
          newD = np.matmul(C[:self.nbf,:self.nel[0]],C.T[:self.nel[0],:self.nbf])
          oneE = 2. * np.trace(np.matmul((newD),(self.T+self.Vne)))
          twoE = 1. * np.trace(np.matmul(newD,(2.*J - K)))
          newE = oneE + twoE
          dE   = abs(newE - energy)
          if options.diis is True:
              dD = e_rms
          else:
              dD = abs(np.sqrt(np.trace((newD-D)**2)))
          energy = newE
          D = 1.*newD
          if it > 0:
              print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,is_diis),flush=True)

          if (dE < options.e_conv) and (dD < options.d_conv):
            if self.options.relativistic == "zora":
              Ci = np.zeros((1,self.nbf),dtype=complex)
              zora_scal = 0.
              for i in range(self.nel[0]):
                Ci[0] = C.T[i]
                Di = np.matmul(np.conjugate(Ci.T),Ci).real
                eps_scal = np.trace(np.matmul(eps_scaling,Di))
                self.eps[i] = evals[i]/(1.+eps_scal)
                zora_scal -= eps_scal*self.eps[i]

              self.eps[self.nel[0]:] = evals[self.nel[0]:]
            else:
              self.eps = evals

              print("    Iterations have converged!")
              print("")
              print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
              print("    One-electron Energy:          %20.12f" %(oneE))
              print("    Two-electron Energy:          %20.12f" %(twoE))
              print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
              print("")
              print("    Orbital Energies [Eh]")
              print("    ---------------------")
              print("")
              print("    Alpha occupied:")
              for i in range(self.nel[0]):
                  print("    %4i: %12.5f "%(i+1,self.eps[i]),end="")
                  if (i+1)%3 == 0: print("")
              print("")
              print("    Alpha Virtual:")
              for a in range(self.nbf-self.nel[0]):
                  print("    %4i: %12.5f "%(self.nel[0]+a+1,self.eps[self.nel[0]+a]),end="")
                  if (a+1)%3 == 0: print("")
              print("")
              break
          if (it == options.maxiter-1):
              print("SCF iterations did not converge!")
          it += 1
      print("")

      print("    Molecular Orbital Analysis")
      print("    --------------------------\n")
  
      ao_labels = self.ints_factory.ao_labels()
      for p in range(self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (2 if p < self.nel[0] else 0),self.eps[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(C.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],C[i][p]),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      self.F = [F]
      self.C = [C]
      self.D = [D]
      self.eps = [self.eps]
      self.scf_energy = twoE + oneE + self.e_nuc

      print("    == Energetics Summary ==")
      print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
      print("    One-electron Energy:          %20.12f" %(oneE))
      print("    Two-electron Energy:          %20.12f" %(twoE))
      print("    @RHF Final Energy: %20.12f" %(self.scf_energy))

      #quick spin analysis 
      N = np.trace((D@self.S))
      self.S2 = N - np.trace((D@self.S)@(D@self.S))
      print("")
      print("    Computed <S2> : %12.8f "%(self.S2))
      print("    Expected <S2> : %12.8f "%(0.0))
           

      return self.scf_energy

class UHF():

    def __init__(self,mol):

      self.ints_factory = mol
      self.nbf          = mol.nao
      self.nel          = np.asarray(mol.nelec,dtype=int) #returns [nalpha, nbeta]
      self.mult         = mol.multiplicity
      self.charge       = mol.charge
      self.nbf          = mol.nao
      self.e_nuc        = mol.energy_nuc() 
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.reference    = "uhf"

   
    def compute(self,options):
      print("    Alpha electrons  : %4i"%(self.nel[0]))
      print("    Beta  electrons  : %5i"%(self.nel[1]))

      self.options = options
      self.S   = self.ints_factory.intor('int1e_ovlp')
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')

      self.jk = scf.UHF(self.ints_factory) #jk object from pyscf

      Fa = self.T + self.Vne
      Fb = self.T + self.Vne

      #orthogonalization matrix
      Shalf = sp.linalg.sqrtm(np.linalg.inv(self.S))
      Faorth = np.matmul(Shalf,np.matmul(Fa,Shalf))
      Fborth = np.matmul(Shalf,np.matmul(Fb,Shalf))

      #find trial movecs
      D            = self.jk.init_guess_by_atom()

      newE = np.einsum("mn,mn->",D[0],(Fa))
      newE += np.einsum("mn,mn->",D[1],(Fb))
      energy = 0
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ")

      if options.diis is True:
        err_veca = np.zeros((1,1,1))
        err_vecb = np.zeros((1,1,1))
        Fdiisa = np.zeros((1,1,1))
        Fdiisb = np.zeros((1,1,1))

      for it in range(options.maxiter):
        Jaa, Kaa = self.jk.get_jk(dm=D[0])
        Jbb, Kbb = self.jk.get_jk(dm=D[1])
        Jab, _ = self.jk.get_jk(dm=D[1])
        Jba, _ = self.jk.get_jk(dm=D[0])
        Ka = Kaa
        Kb = Kbb
        Ja = Jaa + Jab
        Jb = Jbb + Jba
        Fa = self.T + self.Vne + Ja - Ka
        Fb = self.T + self.Vne + Jb - Kb
        if options.diis is True:
          Faorth, err_veca, e_rmsa, Fdiisa, is_diis = diis_routine.do_diis(Fa,D[0],self.S,Shalf,np.asarray(err_veca),Fdiisa)
          Fborth, err_vecb, e_rmsb, Fdiisb, is_diis = diis_routine.do_diis(Fb,D[1],self.S,Shalf,np.asarray(err_vecb),Fdiisb)
        else:
          options.diis = False
          Faorth = np.matmul(Shalf.T,np.matmul(Fa,Shalf))
          Fborth = np.matmul(Shalf.T,np.matmul(Fb,Shalf))
        evals_a, evecs_a = np.linalg.eigh(Faorth)
        Cmo_a = np.matmul(Shalf,evecs_a).real
        newDa = np.matmul(Cmo_a[:,:self.nel[0]],Cmo_a.T[:self.nel[0],:])
        evals_b, evecs_b = np.linalg.eigh(Fborth)
        Cmo_b = np.matmul(Shalf,evecs_b).real
        newDb = np.matmul(Cmo_b[:,:self.nel[1]],Cmo_b.T[:self.nel[1],:])

        oneE = np.trace(np.matmul((newDa+newDb),(self.T+self.Vne)))

        twoE  = 0.5*np.trace(np.matmul(newDa,(Ja-Ka)))
        twoE += 0.5*np.trace(np.matmul(newDb,(Jb-Kb)))
        newE = oneE + twoE
        dE = abs(newE - energy)
        if options.diis is True:
           dD = 0.5 * (abs(e_rmsa) + abs(e_rmsb))
        else:
           dD = 0.5*(abs(np.sqrt(np.trace(newDa-D[0])**2)) + abs(np.sqrt(np.trace(newDb-D[1])**2)))
        energy = newE
        D[0] = 1.*newDa
        D[1] = 1.*newDb
        if it > -1:
            print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,is_diis),flush=True)
        if (dE < options.e_conv) and (dD < options.d_conv):
          print("    Iterations have converged!")
          print("")
          print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
          print("    One-electron Energy:          %20.12f" %(oneE))
          print("    Two-electron Energy:          %20.12f" %(twoE))
          print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
          print("")
          print("    Orbital Energies [Eh]")
          print("    ---------------------")
          print("")
          print("    Alpha occupied:")
          for i in range(self.nel[0]):
              print("    %4i: %12.5f "%(i+1,evals_a[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Alpha Virtual:")
          for a in range(self.nbf-self.nel[0]):
              print("    %4i: %12.5f "%(self.nel[0]+a+1,evals_a[self.nel[0]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          print("")
          print("    Beta occupied:")
          for i in range(self.nel[1]):
              print("    %4i: %12.5f "%(i+1,evals_b[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Beta Virtual:")
          for a in range(self.nbf-self.nel[1]):
              print("    %4i: %12.5f "%(self.nel[1]+a+1,evals_b[self.nel[1]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          break
        if (it == options.maxiter-1):
            print("SCF iterations did not converge!")
        it += 1
      print("")

      print("    Molecular Orbital Analysis")
      print("    --------------------------\n")

      ao_labels = self.ints_factory.ao_labels()
      for p in range(self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (2 if p < self.nel[0] else 0),evals_a[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(Cmo_a.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],Cmo_a[i][p]),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      self.F = [Fa,Fb]
      self.C = [Cmo_a,Cmo_b]
      self.Shalf = Shalf
      self.D = D
      self.eps = [evals_a,evals_b]
      self.scf_energy = twoE + oneE + self.e_nuc

      print("    @UHF Final Energy: %20.12f" %(self.scf_energy))

      #quick spin analysis
      Na = np.trace((D[0]@self.S))
      Nb = np.trace((D[1]@self.S))
      self.S2 = 0.5 * (Na+Nb) + 0.25*(Na-Nb)**2 - np.trace((D[0]@self.S)@(D[1]@self.S))
      print("")
      print("    Computed <S2> : %12.8f "%(self.S2))
      if Nb > Na: 
        print("    Expected <S2> : %12.8f "%((0.5*(Nb-Na))**2+(0.5*(Nb-Na))))
      else: 
        print("    Expected <S2> : %12.8f "%((0.5*(Na-Nb))**2+(0.5*(Na-Nb))))

      return self.scf_energy

class RKS():

    def __init__(self,mol):
      
      self.ints_factory = mol
      self.nbf          = mol.nao 
      self.nel          = mol.nelec #returns [nalpha, nbeta]
      self.mult         = mol.multiplicity 
      self.charge       = mol.charge 
      self.nbf		= mol.nao
      self.e_nuc	= mol.energy_nuc()
      self.reference    = "rks"
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.triplet      = False

    def compute(self,options):
      print("    Alpha electrons  : %4i"%(self.nel[0]),flush=True)
      print("    Beta  electrons  : %5i"%(self.nel[1]),flush=True)
    
      self.options = options

      self.S   = self.ints_factory.intor('int1e_ovlp') 
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')

      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("\n    Exchange-Correlation Functional:",options.xc)
      print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("\n    Hybrid alpha parameter: %f"%options.xcalpha)
      #dft grid parameters
      self.jk    = dft.RKS(self.ints_factory,options.xc) #jk object from pyscf
      #self.jk.multiplicity=2

      eps_scaling = np.zeros((self.nbf,self.nbf))
      self.eps = np.zeros((self.nbf))

      if self.options.relativistic == "zora":
        zora = relativistic.ZORA(self)
        zora.get_zora_correction()
        self.T =  zora.T[0]
        self.H_so = zora.H_so 
        eps_scaling = zora.eps_scal_ao
        if self.options.so_scf is True:
          exit("    CAN'T ADD H_SO TO RKS")

      if options.guess_mos_provided is True:
        print("    Reading MOs from File...")
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
        print("    Computing MOs from PySCF....")
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
        print("    COSMO solvent enabled. Dielectric constant %f"%(self.options.cosmo_epsilon))

      energy = 0.
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ",flush=True)
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
            print("    NOSCF flag enabled. Guess orbitals will be used without optimization")
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
                print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,diis.is_diis),flush=True)

          if (dE < options.e_conv) and (dD < options.d_conv):
            if self.options.relativistic == "zora":
              Ci = np.zeros((1,self.nbf),dtype=complex)
              zora_scal = 0.
              for i in range(self.nel[0]):
                Ci[0] = C.T[i]
                Di = np.matmul(np.conjugate(Ci.T),Ci).real
                eps_scal = np.trace(np.matmul(eps_scaling,Di))
                self.eps[i] = evals[i]/(1.+eps_scal)
                zora_scal -= eps_scal*self.eps[i]
             
              self.eps[self.nel[0]:] = evals[self.nel[0]:]
            else:
              self.eps = evals  

            print("    Iterations have converged!")
            print("")
            print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
            print("    One-electron Energy:          %20.12f" %(oneE))
            print("    Two-electron Energy:          %20.12f" %(twoE))
            print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
            print("")
            print("    Orbital Energies [Eh]")
            print("    ---------------------")
            print("")
            print("    Alpha occupied:")
            for i in range(self.nel[0]):
                print("    %4i: %12.5f "%(i+1,self.eps[i]),end="")
                if (i+1)%3 == 0: print("")
            print("")
            print("    Alpha Virtual:")
            for a in range(self.nbf-self.nel[0]):
                print("    %4i: %12.5f "%(self.nel[0]+a+1,self.eps[self.nel[0]+a]),end="")
                if (a+1)%3 == 0: print("")
            print("")
            break
          if (it == options.maxiter-1):
              print("SCF iterations did not converge!")
              exit(0)
          it += 1
      print("")

      print("    Molecular Orbital Analysis")
      print("    --------------------------\n")
  
      ao_labels = self.ints_factory.ao_labels()
      for p in range(self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (2 if p < self.nel[0] else 0),self.eps[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(C.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],C[i][p]),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      self.F = [F,F]
      self.C = [C,C]
      self.D = [D,D]
      self.eps = [self.eps,self.eps]
      self.scf_energy = twoE + oneE + self.e_nuc
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([C,C],self.nel,mos_filename)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      print("\n")
      print("    Mulliken Population Analysis (q_A = Z_A - Q_A)")
      print("    --------------------------------------------\n")
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
          print("    %-12s: % 8.5f "%(A[a],ZA[a]-QA[a]))
      print("\n")
      print("\n    Energetics Summary")
      print("    ------------------\n")
      print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
      print("    One-electron Energy:          %20.12f" %(oneE))
      print("    Two-electron Energy:          %20.12f" %(twoE))
      print("    @RKS Final Energy:            %20.12f" %(self.scf_energy))

      #quick spin analysis 
      N = np.trace((D@self.S))
      self.S2 = N - np.trace((D@self.S)@(D@self.S))
      print("")
      print("    Computed <S2> : %12.8f "%(self.S2))
      print("    Expected <S2> : %12.8f "%(0.0))
           

      return self.scf_energy

class UKS():

    def __init__(self,mol):

      self.ints_factory = mol
      self.nbf          = mol.nao
      self.nel          = np.asarray(mol.nelec,dtype=int) #returns [nalpha, nbeta]
      self.mult         = mol.multiplicity
      self.charge       = mol.charge
      self.nbf          = mol.nao
      self.e_nuc        = mol.energy_nuc() 
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.reference    = "uks"
      
    def compute(self,options):
      print("    Alpha electrons  : %4i"%(self.nel[0]),flush=True)
      print("    Beta  electrons  : %5i"%(self.nel[1]),flush=True)

      self.options = options
      self.S   = self.ints_factory.intor('int1e_ovlp')
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')

 
      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("\n    Exchange-Correlation Functional:",options.xc)
      print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("\n    Hybrid alpha parameter: %f"%options.xcalpha,flush=True)
      eps_scaling = np.zeros((self.nbf,self.nbf))

      eps_a = np.zeros((self.nbf))
      eps_b = np.zeros((self.nbf))

      #dft grid parameters
      self.jk    = dft.UKS(self.ints_factory,options.xc) #jk object from pyscf

      if self.options.relativistic == "zora":
        zora = relativistic.ZORA(self)
        zora.get_zora_correction()
        self.T =  zora.T[0]
        eps_scaling = zora.eps_scal_ao
        if self.options.so_scf is True:
          exit("    CAN'T ADD H_SO TO UHF")
      Fa =  self.Vne + self.T 
      Fb =  self.Vne + self.T 

      if options.guess_mos_provided is True:
        try:
          print("    Reading MOs from File...")
          Cmo, nel = io_utils.read_real_mos(options.guess_mos)
          Ca = Cmo[0]
          Cb = Cmo[1]
          #sanity check
          if Cmo.shape != (2,self.nbf,self.nbf):
            exit("Incompatible MOs dimension")
          Da = np.matmul(Ca[:,:nel[0]],np.conjugate(Ca.T[:nel[0],:]))
          Db = np.matmul(Cb[:,:nel[1]],np.conjugate(Cb.T[:nel[1],:]))
          D = [Da,Db]
        except:
          Cmo = np.zeros((2*self.nbf,2*self.nbf))
          print("    Could not read MOs. Computing MOs from PySCF....")
          D =  self.jk.init_guess_by_atom().real
      else:
        self.jk.conv_tol	= options.e_conv
        self.jk.conv_tol_grad = options.d_conv
        self.jk.level_shift = 0. 
        self.jk.max_cycle = options.maxiter
        self.jk.verbose = 4

        
        energy_pyscf = self.jk.kernel()
        Cmo = np.zeros((2*self.nbf,2*self.nbf))
        D   = np.zeros((2,self.nbf,self.nbf))
        print("    Computing MOs from PySCF....",flush=True)
        C =  self.jk.mo_coeff
        Ca =  C[0]
        Cb =  C[1]
        D[0] = Ca[:,:self.nel[0]]@Ca.T[:self.nel[0],:]
        D[1] = Cb[:,:self.nel[1]]@Cb.T[:self.nel[1],:]

      #orthogonalization matrix
      Shalf = sp.linalg.sqrtm(np.linalg.inv(self.S))

      if self.options.cosmo is True: 
        cosmo = solvent.ddcosmo.DDCOSMO(self.ints_factory)
        cosmo.eps = self.options.cosmo_epsilon 
        print("    COSMO solvent enabled. Dielectric constant %f"%(self.options.cosmo_epsilon))

      pot = self.jk.get_veff(self.ints_factory,(D[0],D[1])) #pyscf effective potential
      newE  = np.einsum("mn,mn->",D[0],(Fa)) 
      newE += np.einsum("mn,mn->",D[1],(Fb))
      energy = newE+pot.ecoul+pot.exc
#      print(energy)
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]| ",flush=True)

      if options.diis is True:
        err_veca = np.zeros((1,1,1))
        err_vecb = np.zeros((1,1,1))
        Fdiisa = np.zeros((1,1,1))
        Fdiisb = np.zeros((1,1,1))
        diisa = diis_routine.DIIS(self)
        diisb = diis_routine.DIIS(self)

      for it in range(options.maxiter):
        pot = self.jk.get_veff(self.ints_factory,(D[0],D[1])) #pyscf effective potential
         
        Fa = self.T + self.Vne + pot[0]
        Fb = self.T + self.Vne + pot[1]
        if self.options.cosmo is True:
          e_cosmo, v_cosmo = cosmo.kernel(dm=D)
          Fa += v_cosmo
          Fb += v_cosmo

        if options.diis is True:
          Faorth, e_rmsa = diisa.do_diis(Fa,D[0],self.S,Shalf)
          Fborth, e_rmsb = diisb.do_diis(Fb,D[1],self.S,Shalf)
        else:
          options.diis = False
          Faorth = np.matmul(Shalf.T,np.matmul(Fa,Shalf))
          Fborth = np.matmul(Shalf.T,np.matmul(Fb,Shalf))
        evals_a, evecs_a = sp.linalg.eigh(Faorth)
        Cmo_a = np.matmul(Shalf,evecs_a).real
        evals_b, evecs_b = sp.linalg.eigh(Fborth)
        Cmo_b = np.matmul(Shalf,evecs_b).real
        if options.noscf is True:
          print("    NOSCF flag enabled. Guess orbitals will be used without optimization")
          dE = options.e_conv/10.
          dD = options.d_conv/10.
          oneE = np.trace(np.matmul((D[0]+D[1]),(self.T+self.Vne)))
          if self.options.cosmo is True:
            oneE  += e_cosmo 
          twoE  = pot.ecoul + pot.exc 
          newE  = oneE + twoE
          energy = newE
        else:
          newDa = np.matmul(Cmo_a[:,:self.nel[0]],Cmo_a.T[:self.nel[0],:])
          newDb = np.matmul(Cmo_b[:,:self.nel[1]],Cmo_b.T[:self.nel[1],:])

          oneE = np.trace(np.matmul((newDa+newDb),(self.T+self.Vne)))

          twoE = pot.ecoul + pot.exc
          newE = oneE + twoE
          dE = abs(newE - energy)
          if options.diis is True:
             dD = 0.5 * (abs(e_rmsa) + abs(e_rmsb))
          else:
             dD = 0.5*(abs(np.sqrt(np.trace(newDa-D[0])**2)) + abs(np.sqrt(np.trace(newDb-D[1])**2)))
          energy = newE
          D[0] = 1.*newDa
          D[1] = 1.*newDb
          if it > -1:
              print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s"%(it,energy+self.e_nuc,dE,dD,diisa.is_diis),flush=True)
        if (dE < options.e_conv) and (dD < options.d_conv):
          if self.options.relativistic == "zora":
            Ci = np.zeros((1,self.nbf),dtype=complex)
            zora_scal = 0.
            for i in range(self.nel[0]):
              Ci[0] = Cmo_a.T[i]
              Di = np.matmul(np.conjugate(Ci.T),Ci).real
              eps_scal = np.trace(np.matmul(eps_scaling,Di))
              eps_a[i] = evals_a[i]/(1.+eps_scal)
              zora_scal -= eps_scal*eps_a[i]
            for i in range(self.nel[1]):
              Ci[0] = Cmo_b.T[i]
              Di = np.matmul(np.conjugate(Ci.T),Ci).real
              eps_scal = np.trace(np.matmul(eps_scaling,Di))
              eps_b[i] = evals_b[i]/(1.+eps_scal)
              zora_scal -= eps_scal*eps_b[i]
            eps_a[self.nel[0]:] = evals_a[self.nel[0]:]
            eps_b[self.nel[1]:] = evals_b[self.nel[1]:]
          else:
            eps_a = evals_a
            eps_b = evals_b
          print("    Iterations have converged!")
          print("")
          print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
          print("    One-electron Energy:          %20.12f" %(oneE))
          print("    Two-electron Energy:          %20.12f" %(twoE))
          if self.options.relativistic == "zora":
            print("    Zora Energy Scaling:        %20.12f" %(zora_scal))
          print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
          print("")
          print("    Orbital Energies [Eh]")
          print("    ---------------------")
          print("")
          print("    Alpha occupied:")
          for i in range(self.nel[0]):
              print("    %4i: %12.5f "%(i+1,eps_a[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Alpha Virtual:")
          for a in range(self.nbf-self.nel[0]):
              print("    %4i: %12.5f "%(self.nel[0]+a+1,eps_a[self.nel[0]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          print("")
          print("    Beta occupied:")
          for i in range(self.nel[1]):
              print("    %4i: %12.5f "%(i+1,eps_b[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Beta Virtual:")
          for a in range(self.nbf-self.nel[1]):
              print("    %4i: %12.5f "%(self.nel[1]+a+1,eps_b[self.nel[1]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          break
        if (it == options.maxiter-1):
            print("SCF iterations did not converge!")
            exit(0)
        it += 1
      print("")

      ao_labels = self.ints_factory.ao_labels()
      print("    Alpha Molecular Orbital Analysis")
      print("    --------------------------------\n")
      for p in range(self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (1 if p < self.nel[0] else 0),eps_a[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(Cmo_a.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],Cmo_a[i][p]),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      print("    Beta Molecular Orbital Analysis")
      print("    -------------------------------\n")
      for p in range(self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (1 if p < self.nel[1] else 0),eps_b[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(Cmo_b.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],Cmo_b[i][p]),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      print("    AO labels")
      for i in range(len(ao_labels)):
        print("    %i: %-12s"%(i+1,ao_labels[i]))

      self.F = [Fa,Fb]
      self.C = [Cmo_a,Cmo_b]
      self.Shalf = Shalf
      self.D = D
      self.eps = [eps_a,eps_b]
      self.scf_energy = twoE + oneE + self.e_nuc
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([Cmo_a,Cmo_b],self.nel,mos_filename)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      print("    @UKS Final Energy:   %20.12f" %(self.scf_energy))

      #quick spin analysis
      Na = np.trace((D[0]@self.S))
      Nb = np.trace((D[1]@self.S))
      self.S2 = 0.5 * (Na+Nb) + 0.25*(Na-Nb)**2 - np.trace((D[0]@self.S)@(D[1]@self.S))
      print("")
      print("    Computed <S2> : %12.8f "%(self.S2))
      if Nb > Na:
        print("    Expected <S2> : %12.8f "%((0.5*(Nb-Na))**2+(0.5*(Nb-Na))))
      else:
        print("    Expected <S2> : %12.8f "%((0.5*(Na-Nb))**2+(0.5*(Na-Nb))))

      return self.scf_energy

class GKS():

    def __init__(self,mol):
      self.ints_factory = mol
      self.nbf          = mol.nao
      self.nel          = [np.sum(np.asarray(mol.nelec,dtype=int)),0] #returns nelectrons
      self.mult         = mol.multiplicity
      self.charge       = mol.charge
      self.nbf          = mol.nao
      self.e_nuc        = mol.energy_nuc() 
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.reference    = "gks"
      
    def compute(self,options):
      print("    Number of electrons        : %4i"%(self.nel[0]))
      print("    Number of basis functions  : %4i"%(2*self.nbf))

      self.options = options
      self.S   = self.ints_factory.intor('int1e_ovlp')
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')

      #integrals for electric and magnetic field perturbations
      self.mu  = self.ints_factory.intor('int1e_r')
      self.mag = 0.5j*self.ints_factory.intor('int1e_giao_irjxp')
     
      self.B_field = self.options.B_field_amplitude 
      if self.B_field > 0.:
        self.B_pol   = self.options.B_field_polarization 
        print("\n    Static Magnetic Field Enabled!")
        print("    B Field Amplitude:    %8.5f a.u."%self.B_field)
        print("    B Field Polarization: %2i"%self.B_pol)
        self.Vne = self.Vne - self.B_field * self.mag[self.B_pol]

      self.E_field = self.options.E_field_amplitude 
      if self.E_field > 0.:
        self.E_pol   = self.options.E_field_polarization 
        print("\n    Static Magnetic Field Enabled!")
        print("    E Field Amplitude:    %8.5f a.u."%self.E_field)
        print("    E Field Polarization: %2i"%self.E_pol)
        self.Vne = self.Vne - self.E_field * self.mu[self.E_pol]

      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("\n    Exchange-Correlation Functional:",options.xc)
      print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("\n    Hybrid alpha parameter: %f"%options.xcalpha)
 
      self.jk    = dft.GKS(self.ints_factory,options.xc) #jk object from pyscf
      #find trial movecs
      if options.guess_mos_provided is True:
        print("    Reading MOs from File...")
        try:
          Cmo, nel = io_utils.read_complex_mos(options.guess_mos)
          if Cmo.shape == (2,2*self.nbf,2*self.nbf):
            print("    GKS orbitals found.",flush=True)  
            D = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))  
          elif Cmo.shape == (2,self.nbf,self.nbf):
            print("    RKS orbitals found. projecting into RGKS space",flush=True)
            D = np.zeros((2*self.nbf,2*self.nbf),dtype=complex)  
            D[:self.nbf,:self.nbf] = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))  
            D[self.nbf:,self.nbf:] = np.matmul(Cmo[1][:,:nel[1]],np.conjugate(Cmo[1].T[:nel[1],:]))  
          else:
            print("    Guess orbitals with wrong dimensions! Computing trial density from PySCF....")
            D =  self.jk.init_guess_by_atom()
        except:
          print("    MOs not found! Computing trial density from PySCF....")
          D =  self.jk.init_guess_by_atom()
      else:
        print("    Failed reading MOs from file! Computing trial density from PySCF....")
        self.jk.conv_tol	= options.e_conv
        self.jk.conv_tol_grad = options.d_conv
        self.jk.verbose = 4
        energy_pyscf = self.jk.kernel()
        C =  self.jk.mo_coeff
        D = C[:,:self.nel[0]]@C.T[:self.nel[0],:].real

      F   = np.zeros((2*self.nbf,2*self.nbf),dtype=complex)
      F0  = np.zeros((2*self.nbf,2*self.nbf),dtype=complex)
      S   = np.zeros((2*self.nbf,2*self.nbf))
      eps_scaling = np.zeros((2*self.nbf,2*self.nbf))

      Cmo = np.zeros((2*self.nbf,2*self.nbf),dtype=complex)
      eps = np.zeros((2*self.nbf))

      if self.options.relativistic == "zora":
        zora = relativistic.ZORA(self)
        zora.get_zora_correction()
        F0[:self.nbf,:self.nbf] =  self.Vne + zora.T[0]
        F0[self.nbf:,self.nbf:] =  self.Vne + zora.T[0]
        eps_scaling[:self.nbf,:self.nbf] = zora.eps_scal_ao
        eps_scaling[self.nbf:,self.nbf:] = zora.eps_scal_ao    
        self.H_so = zora.H_so
        if self.options.so_scf is True:
          F0 += zora.H_so
      else:
        F0[:self.nbf,:self.nbf] =  self.Vne + self.T 
        F0[self.nbf:,self.nbf:] =  self.Vne + self.T 

      S[:self.nbf,:self.nbf] = self.S
      S[self.nbf:,self.nbf:] = self.S
      Shalf = sp.linalg.sqrtm(np.linalg.inv(S))

      mu  = np.zeros((3,2*self.nbf,2*self.nbf))
      mu[0][:self.nbf,:self.nbf] = self.mu[0]
      mu[0][self.nbf:,self.nbf:] = self.mu[0]
      mu[1][:self.nbf,:self.nbf] = self.mu[1]
      mu[1][self.nbf:,self.nbf:] = self.mu[1]
      mu[2][:self.nbf,:self.nbf] = self.mu[2]
      mu[2][self.nbf:,self.nbf:] = self.mu[2]
      

      newE = np.einsum("mn,mn->",D,F0) #+ Exc
      energy = 0
      newD = np.zeros((2*self.nbf,2*self.nbf),dtype=complex)
      print("    Starting SCF procedure.",flush=True) 
      tic = time.time()
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]|  Time(s) ",flush=True)

      if options.diis is True:
        err_vec = np.zeros((1,1,1))
        Fdiis = np.zeros((1,1,1))
        diis = diis_routine.DIIS(self) 
        diis.F_vecs = np.zeros((diis.diis_max,2*self.nbf,2*self.nbf),dtype=complex)
        diis.e_vecs = np.zeros((diis.diis_max,2*self.nbf,2*self.nbf),dtype=complex)

      for it in range(options.maxiter):
        pot = self.jk.get_veff(self.ints_factory,D) #pyscf effective potential
        F = F0 + pot
        Forth, e_rms = diis.do_diis(F,D,S,Shalf)
        evals, evecs = np.linalg.eigh(Forth)
        Cmo   = np.matmul(Shalf,evecs)
        newD  = np.matmul(Cmo[:,:self.nel[0]],np.conjugate(Cmo.T[:self.nel[0],:]))
        oneE  = np.trace(np.matmul(newD,F0)).real
        twoE  = pot.ecoul + pot.exc 
        newE  = oneE + twoE
        dE = np.abs(newE - energy)
        #dD = np.sqrt(e_rms.real**2 + e_rms.imag**2)
        dD = np.sqrt(e_rms.real**2 + e_rms.imag**2)
        energy = newE.real
        D = 1.*newD
        if it > -1:
            print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s %5.2f"%(it,energy+self.e_nuc,dE,dD,diis.is_diis,time.time()-tic),flush=True)
        if (dE < options.e_conv) and (dD < options.d_conv):
          if self.options.relativistic == "zora":
            Ci = np.zeros((1,2*self.nbf),dtype=complex)
            zora_scal = 0.
            for i in range(self.nel[0]):
              Ci[0] = Cmo.T[i]
              Di = np.matmul(np.conjugate(Ci.T),Ci).real
              eps_scal = np.trace(np.matmul(eps_scaling,Di))
              eps[i] = evals[i]/(1.+eps_scal)
              zora_scal -= eps_scal*eps[i]
            eps[self.nel[0]:] = evals[self.nel[0]:]
          else:
            eps = evals


          print("    Iterations have converged!")
          print("")
          print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
          print("    One-electron Energy:          %20.12f" %(oneE))
          print("    Two-electron Energy:          %20.12f" %(twoE))
          if self.options.relativistic == "zora":
            print("    Zora Energy scaling:          %20.12f" %(zora_scal))
          print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
          print("")
          print("    Orbital Energies [Eh]")
          print("    ---------------------")
          print("")
          print("    Occupied orbitals:")
          for i in range(self.nel[0]):
              print("    %4i: %12.5f "%(i+1,eps[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Virtual orbitals:")
          for a in range(2*self.nbf-self.nel[0]):
              print("    %4i: %12.5f "%(self.nel[0]+a+1,eps[self.nel[0]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          break
        if (it == options.maxiter-1):
            print("SCF iterations did not converge!")
        it += 1
      print("")

      print("    Molecular Orbital Analysis")
      print("    --------------------------\n")

      ao_labels = 2*self.ints_factory.ao_labels()
      for p in range(2*self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (1 if p < self.nel[0] else 0),eps[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(Cmo.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],Cmo[i][p].real),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      self.F = [F]
      self.C = [Cmo]
      self.Shalf = [Shalf]
      self.D = [D]
      self.mu = mu
      self.eps = [eps]
      self.scf_energy = twoE + oneE + self.e_nuc
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([Cmo,Cmo],self.nel,mos_filename)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      self.E0 = (energy.real+self.e_nuc)
      print("")
      print("    @GHF Final Energy: %20.12f Eh" %self.E0)
      print("    @GHF Final Energy: %20.12f eV" %(self.E0*27.21138))
#      print("    @PYSCF Final Energy: %20.12f Eh" %(energy_pyscf))
      print("")

      toc = time.time()
      print("    SCF intrations took %5.2f seconds"%(toc-tic),flush=True)

      return self.scf_energy

class RGKS():

    def __init__(self,mol):

      self.ints_factory = mol
      self.nbf          = mol.nao
      self.nel          = [np.sum(np.asarray(mol.nelec,dtype=int)),0] #returns nelectrons
      self.mult         = mol.multiplicity
      self.charge       = mol.charge
      self.nbf          = mol.nao
      self.e_nuc        = mol.energy_nuc() 
      self.molden_reorder = tools.molden.order_ao_index(self.ints_factory)
      self.reference    = "rgks"
      
    def compute(self,options):
      print("    Number of electrons        : %4i"%(self.nel[0]))
      print("    Number of basis functions  : %4i"%(2*self.nbf))

      self.options = options
      self.S   = self.ints_factory.intor('int1e_ovlp')
      self.T   = self.ints_factory.intor('int1e_kin')
      self.Vne = self.ints_factory.intor('int1e_nuc')
      self.mu  = self.ints_factory.intor('int1e_r')
 
      options.xctype  = dft.libxc.xc_type(self.options.xc)
      options.xcalpha = dft.libxc.hybrid_coeff(self.options.xc)
      print("\n    Exchange-Correlation Functional:",options.xc)
      print("\n    Exchange-Correlation Functional Type:",options.xctype)
      print("\n    Hybrid alpha parameter: %f"%options.xcalpha,flush=True)

      self.jk    = dft.GKS(self.ints_factory,options.xc) #jk object from pyscf
      #find trial movecs
      if options.guess_mos_provided is True:
        print("    Reading MOs from File...")
        try:
          Cmo, nel = io_utils.read_real_mos(options.guess_mos)
          if Cmo.shape == (2,2*self.nbf,2*self.nbf):
            print("    RGKS orbitals found.",flush=True)  
            D = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))  
          elif Cmo.shape == (2,self.nbf,self.nbf):
            print("    RKS orbitals found. projecting into RGKS space",flush=True)
            D = np.zeros((2*self.nbf,2*self.nbf))  
            D[:self.nbf,:self.nbf] = np.matmul(Cmo[0][:,:nel[0]],np.conjugate(Cmo[0].T[:nel[0],:]))  
            D[self.nbf:,self.nbf:] = np.matmul(Cmo[1][:,:nel[1]],np.conjugate(Cmo[1].T[:nel[1],:]))  
          else:
            print("    Guess orbitals with wrong dimensions! Computing trial density from PySCF....")
            D =  self.jk.init_guess_by_atom().real
        except:
          print("    MOs not found! Computing trial density from PySCF....")
          D =  self.jk.init_guess_by_atom().real
      else:
        print("    Failed reading MOs from file! Computing trial density from PySCF....")
        self.jk.conv_tol	= options.e_conv
        self.jk.conv_tol_grad = options.d_conv
        self.jk.verbose = 4
        energy_pyscf = self.jk.kernel()
        D =  self.jk.make_rdm1().real

      F   = np.zeros((2*self.nbf,2*self.nbf))
      F0  = np.zeros((2*self.nbf,2*self.nbf))
      S   = np.zeros((2*self.nbf,2*self.nbf))
      eps_scaling = np.zeros((2*self.nbf,2*self.nbf))

      eps = np.zeros((2*self.nbf))

      if self.options.relativistic == "zora":
        zora = relativistic.ZORA(self)
        zora.get_zora_correction()
        F0[:self.nbf,:self.nbf] =  self.Vne + zora.T[0]
        F0[self.nbf:,self.nbf:] =  self.Vne + zora.T[0]
        eps_scaling[:self.nbf,:self.nbf] = zora.eps_scal_ao
        eps_scaling[self.nbf:,self.nbf:] = zora.eps_scal_ao    
        self.H_so = zora.H_so
        if self.options.so_scf is True:
          exit("    ERROR: Cannot add Hso to Real GHF reference!")
      else:
        F0[:self.nbf,:self.nbf] =  self.Vne + self.T 
        F0[self.nbf:,self.nbf:] =  self.Vne + self.T 

      S[:self.nbf,:self.nbf] = self.S
      S[self.nbf:,self.nbf:] = self.S

      mu  = np.zeros((3,2*self.nbf,2*self.nbf))
      mu[0][:self.nbf,:self.nbf] = self.mu[0]
      mu[0][self.nbf:,self.nbf:] = self.mu[0]
      mu[1][:self.nbf,:self.nbf] = self.mu[1]
      mu[1][self.nbf:,self.nbf:] = self.mu[1]
      mu[2][:self.nbf,:self.nbf] = self.mu[2]
      mu[2][self.nbf:,self.nbf:] = self.mu[2]
      Shalf = sp.linalg.sqrtm(np.linalg.inv(S))

      newE = np.einsum("mn,mn->",D,F0) #+ Exc
      energy = 0
      newD = np.zeros((2*self.nbf,2*self.nbf))
      if self.options.cosmo is True: 
        cosmo = solvent.ddcosmo.DDCOSMO(self.ints_factory)
        cosmo.eps = self.options.cosmo_epsilon 
        print("    COSMO solvent enabled. Dielectric constant %f"%(self.options.cosmo_epsilon))

      print("    Starting SCF procedure.",flush=True) 
      tic = time.time()
      print("\n                   Total Energy    |Delta E|    RMS |[F,D]|  Time(s) ",flush=True)

      if options.diis is True:
        print("Initializing DIIS",flush=True)
        err_vec = np.zeros((1,1,1))
        Fdiis = np.zeros((1,1,1))
        diis = diis_routine.DIIS(self) 
        diis.F_vecs = np.zeros((diis.diis_max,2*self.nbf,2*self.nbf))
        diis.e_vecs = np.zeros((diis.diis_max,2*self.nbf,2*self.nbf))

      for it in range(options.maxiter):
        print("Evaluating effective potential",flush=True)
        pot = self.jk.get_veff(self.ints_factory,D) #pyscf effective potential
        F = F0 + pot.real
        if self.options.cosmo is True:
          Daa = D[:self.nbf,:self.nbf]
          Dbb = D[self.nbf:,self.nbf:]
          e_cosmo, v_cosmo = cosmo.kernel(dm=[Daa,Dbb])
          F[:self.nbf,:self.nbf] += v_cosmo
          F[self.nbf:,self.nbf:] += v_cosmo
        print("Extrapolating Fock matrix",flush=True)
        Forth, e_rms = diis.do_diis(F,D,S,Shalf)
        print("Diagonalizing Fock matrix",flush=True)
        evals, evecs = sp.linalg.eigh(Forth)
        print("Back-transforming MOs",flush=True)
        Cmo   = Shalf@evecs.real
 
        if options.noscf is True:
          print("    NOSCF flag enabled. Guess orbitals will be used without optimization",flush=True)
          if len(options.swap_mos) == 2:
            orig = options.swap_mos[0]
            swap = options.swap_mos[1]
            for i in range(len(orig)):
              print("    WARNING: Swapping molecular orbitals %i and %i"%(orig[i]+1,swap[i]+1))
              orig_i = [orig[i],swap[i]]
              swap_i = [swap[i],orig[i]]
              Cmo[:,swap_i] = Cmo[:,orig_i]
              evals[swap_i[0]], evals[swap_i[1]] = evals[orig_i[0]], evals[orig_i[1]]

          dE = options.e_conv/10.
          dD = options.d_conv/10.
          oneE  = np.einsum("pq,pq->",D,F0,optimize=True).real 
          if self.options.cosmo is True:
            oneE  += e_cosmo 
          twoE  = pot.ecoul + pot.exc 
          newE  = oneE + twoE
          energy = newE
        else:
          newD  = np.matmul(Cmo[:,:self.nel[0]],Cmo.T[:self.nel[0],:])
          oneE  = np.trace(np.matmul(newD,F0)) 
          if self.options.cosmo is True:
            oneE  += e_cosmo 
          twoE  = pot.ecoul + pot.exc 
          newE  = oneE + twoE
          dE = np.abs(newE - energy)
          dD = np.abs(e_rms)
          energy = newE
          D = 1.*newD
          if it > 0:
              print("    @SCF iter %3i: % 12.8f % 10.6e % 10.6e %6s %5.2f"%(it,energy+self.e_nuc,dE,dD,diis.is_diis,time.time()-tic),flush=True)
        if (dE < options.e_conv) and (dD < options.d_conv):
          if self.options.relativistic == "zora":
            Ci = np.zeros((1,2*self.nbf),dtype=complex)
            zora_scal = 0.
            for i in range(self.nel[0]):
              Ci[0] = Cmo.T[i]
              Di = np.conjugate(Ci.T)@Ci
              eps_scal = np.trace(eps_scaling@Di)
              eps[i] = evals[i]/(1.+eps_scal)
              zora_scal -= eps_scal*eps[i]
            eps[self.nel[0]:] = evals[self.nel[0]:]
          else:
            eps = evals


          print("    Iterations have converged!")
          print("")
          print("    Nuclear Repulsion Energy:     %20.12f" %(self.e_nuc))
          print("    One-electron Energy:          %20.12f" %(oneE))
          print("    Two-electron Energy:          %20.12f" %(twoE))
          if self.options.relativistic == "zora":
            print("    Zora Energy scaling:          %20.12f" %(zora_scal)) 
          if self.options.cosmo is True:
            print("    Cosmo Energy:               %20.12f" %(e_cosmo)) 
          print("    Total  Energy:                %20.12f\n" %(twoE+oneE+self.e_nuc))
          print("")
          print("    Orbital Energies [Eh]")
          print("    ---------------------")
          print("")
          print("    Occupied orbitals:")
          for i in range(self.nel[0]):
              print("    %4i: %12.5f "%(i+1,eps[i]),end="")
              if (i+1)%3 == 0: print("")
          print("")
          print("    Virtual orbitals:")
          for a in range(2*self.nbf-self.nel[0]):
              print("    %4i: %12.5f "%(self.nel[0]+a+1,eps[self.nel[0]+a]),end="")
              if (a+1)%3 == 0: print("")
          print("")
          break
        if (it == options.maxiter-1):
            print("SCF iterations did not converge!")
        it += 1
      print("")

      print("    Molecular Orbital Analysis")
      print("    --------------------------\n")

      ao_labels = 2*self.ints_factory.ao_labels()
      for p in range(2*self.nbf):
        print("    Vector %5i:    Occupation = %1i    Energy = %8.5f "%(p+1, (1 if p < self.nel[0] else 0),eps[p]))
        print("    ----------------------------------------------------------------------------")
        sort_idx = np.argsort(-np.abs(Cmo.T[p]))
        for idx, i in enumerate(sort_idx[:6]):
          print("    %-12s: % 8.5f "%(ao_labels[i],Cmo[i][p].real),end="")
          if ((idx+1)%3 == 0): print("")
        print("\n")

      self.F = [F]
      self.C = [Cmo]
      self.Shalf = Shalf
      self.D = [D]
      self.eps = [eps]
      self.mu = mu
      self.scf_energy = twoE + oneE + self.e_nuc
      print("Writing MOs",flush=True)
      mos_filename = self.options.inputfile.split(".")[0]+".mos"
      io_utils.write_mos([Cmo,Cmo],self.nel,mos_filename)
      print("Writing Molden file",flush=True)
      filename = self.options.inputfile.split(".")[0]
      io_utils.write_molden(self,filename)

      self.E0 = (energy.real+self.e_nuc)
      print("")
      print("    @RGHF Final Energy: %20.12f Eh" %self.E0)
      print("    @RGHF Final Energy: %20.12f eV" %(self.E0*27.21138))
      print("")

      toc = time.time()
      print("    SCF intrations took %5.2f seconds"%(toc-tic),flush=True)

      return self.scf_energy
