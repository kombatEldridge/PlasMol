import os
import numpy as np
from scipy.linalg import inv
from pyscf import scf, dft, solvent, tools, gto
from pyscf.scf import addons

from bohr_internals import diis_routine
from bohr_internals import io_utils
from bohr_internals import constants
from chkfile import restart_from_chkfile
'''
Here, scf is rhf
    mol = gto.M(...)
    rhf = scf.RHF(mol)
    rhf.kernel()
'''
class SCF():
    def __init__(self, scf, chkfile=None):
        self._scf = scf
        self.S = self._scf.get_ovlp()
        self.X = addons.canonical_orth_(self.S)
        self.C = self._scf.mo_coeff
        self.occ = self._scf.get_occ()

        self.D_ao_0 = self._scf.make_rdm1(mo_occ=self.occ)

        if len(np.shape(self.D_ao_0)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        self.start_time = 0
        # self.chkfile = chkfile
        # if chkfile is not None and os.path.exists(chkfile):
        #     self.start_time = restart_from_chkfile(self)
        #     self.D_ao_0 = self._scf.make_rdm1(mo_occ=self.occ)

