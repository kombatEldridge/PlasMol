import bohr_internals.input_parser as input_parser
from pyscf import gto
import bohr_internals.wavefunction as wavefunction
import numpy as np
import logging
import matrix_handler as mh

class MOLECULE():
    def __init__(self, inputfile):
        import bohr_internals.options as options
        options = options.OPTIONS()

        self.molecule, self.method, basis = input_parser.read_input(inputfile, options)
        options.molecule = self.molecule

        #Format molecule string as required by PySCF
        atoms = self.molecule["atoms"]
        pyscf_molecule = "" 
        for index, atom in enumerate(atoms):
            pyscf_molecule += " " + atom
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][0])
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][1])
            pyscf_molecule += " " + str(self.molecule["coords"][atom+str(index+1)][2])
            if index != (len(atoms)-1):
                pyscf_molecule += ";"

        # Added B3LYP_WITH_VWN5 = True
        # to /Users/bldrdge1/.conda/envs/meep/lib/python3.11/site-packages/pyscf/__config__.py
        pyscf_mol = gto.M(atom = pyscf_molecule, 
                        basis  = basis["name"], 
                        unit   = 'B', 
                        charge = int(options.charge), 
                        spin   = int(options.spin), 
                        cart   = options.cartesian)
        pyscf_mol.set_common_origin(self.molecule["com"])
        pyscf_mol.verbose = 0
        pyscf_mol.max_memory = options.memory
        pyscf_mol.build()

        self.wfn = wavefunction.RKS(pyscf_mol)
        rks_energy = self.wfn.compute(options)
        
        D_ao_0 = self.wfn.D[0]
        self.D_mo_0 = self.wfn.C[0].T @ self.wfn.S @ D_ao_0 @ self.wfn.S @ self.wfn.C[0]
        trace = np.trace(self.D_mo_0)
        n = self.wfn.nel[0]
        if not np.isclose(trace, n):
            raise ValueError(f"Trace of the matrix is not {n} (instead {trace}).")
        mh.set_D_mo_0(self.D_mo_0)

        F_ao_0 = self.wfn.F[0]
        F_mo_0 = self.wfn.S @ self.wfn.C[0] @ F_ao_0 @ self.wfn.C[0].T @ self.wfn.S 
        mh.set_F_mo_0(F_mo_0)
    