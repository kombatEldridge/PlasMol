# quantum/induced_mu.py
import numpy as np

def induced_dipole(molecule):
    """
    Calculate the induced dipole moment of the molecule.

    Parameters:
    molecule : object
        Molecule object with current state.

    Returns:
    array-like
        Induced dipole moment.
    """
    mu = molecule.calculate_mu()

    D_AO = molecule.D_ao
    D0_AO = molecule.D_ao_0

    if D_AO.ndim == 3:
        D_AO = D_AO.sum(axis=0)
        D0_AO = D0_AO.sum(axis=0)

    mu_arr = np.zeros(3)
    for i in range(3):
        mu_arr[i] = float((np.trace(mu[i] @ D_AO) - np.trace(mu[i] @ D0_AO)).real)

    return mu_arr

def induced_dipole_using_mo(molecule):
    """
    Calculate the induced dipole moment of the molecule using MO basis.
    This should be the same as ind dip using AO basis.

    Parameters:
    molecule : object
        Molecule object with current state.

    Returns:
    array-like
        Induced dipole moment in MO basis.
    """
    mu_mo = calculate_mu_mo(molecule)

    D_MO = molecule.neutral_mo_density()
    D0_MO = molecule.neutral_mo_density(D0)
    
    mu_arr = np.zeros(3)
    if molecule.is_open_shell:
        for i in range(3):
            now = np.trace(mu_mo[0, i] @ D_MO[0]) + np.trace(mu_mo[1, i] @ D_MO[1])
            ref = np.trace(mu_mo[0, i] @ D0_MO[0]) + np.trace(mu_mo[1, i] @ D0_MO[1])
            mu_arr[i] = float((now - ref).real)
    else:
        for i in range(3):
            mu_arr[i] = float((np.trace(mu_mo[i] @ D_MO) - np.trace(mu_mo[i] @ D0_MO)).real)    

    return mu_arr

def calculate_mu_mo(molecule):
    """
    Calculate the dipole moment integrals for the molecule.

    Sets the origin to the nuclear charge center and computes dipole integrals.

    Parameters:
    None

    Returns:
    np.ndarray
        Dipole moment integrals with shape (3, nao, nao) for x, y, z components.
    """
    mu = -1 * molecule.mf.mol.intor('int1e_r', comp=3)

    def project(C):
        Ct = C.conj().T
        mu_mo = np.empty((3, C.shape[1], C.shape[1]), dtype=np.result_type(mu, C))
        for i in range(3):
            mu_mo[i] = Ct @ mu[i] @ C
        return mu_mo

    if molecule.is_open_shell:
        return np.stack((project(molecule.C[0]), project(molecule.C[1])))
    return project(molecule.C)