# params.py
class PARAMS:
    """
    Class to hold parameters for the quantum simulation.

    This class encapsulates the configuration settings for simulating the dynamics
    of a molecular system under an external field. It includes the molecular structure,
    the applied field, numerical tolerances, and simulation method specifications.

    Parameters
    ----------
    pcconv : float, optional
        Tolerance for the convergence  used in the propagator. Default is 1e-12.
    tol_zero : float, optional
        Tolerance for numerical checks, such as verifying if matrices are Hermitian
        or unitary. This is distinct from the convergence tolerance used in the
        propagator. Default is 1e-12.
    doublecheck : bool, optional
        If True, performs checks to ensure the Fock matrix is Hermitian and the
        time evolution operator (U matrix) is unitary. Default is True.
    exp_method : int, optional
        Method used for computing the matrix exponential in the time propagation:
        - 1: Series expansion method
        - 2: Diagonalization method
        Default is 1.
    dt : float, optional
        Time step for the simulation, in atomic units (au). Default is 0.01.
    terms_interpol : int, optional
        Number of consecutive iterations required to achieve convergence in the
        interpolation process. Default is 2.

    Attributes
    ----------
    The attributes correspond directly to the parameters provided during initialization.
    """
    def __init__(self, pcconv=1e-12, tol_zero=1e-12, doublecheck=True, exp_method=1, dt=0.01, terms_interpol=2, max_iter=200):
        self.pcconv = pcconv
        self.tol_zero = tol_zero
        self.doublecheck = doublecheck
        self.exp_method = exp_method
        self.dt = dt
        self.terms_interpol = terms_interpol
        self.max_iter = max_iter

