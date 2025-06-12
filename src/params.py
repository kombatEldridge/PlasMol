# params.py
class PARAMS:
    """
    Container for simulation parameters typically set via command-line arguments.
    """
    def __init__(self, pcconv=1e-12, tol_zero=1e-12, dt=0.01, max_iter=200, 
                 chkfile=True, chkfile_freq=10, chkfile_path='chkfile.npz', peak_time_au=1.0, 
                 width_steps=5, shape='kick', smoothing=False, intensity_au=5e-5, 
                 eFieldFile='eField.csv', pFieldFile='pField.csv'):
        """
        Initialize the PARAMS object with simulation parameters.

        Parameters:
        pcconv : float, optional
            Convergence criterion for predictor-corrector (default 1e-12).
        tol_zero : float, optional
            Numerical zero tolerance (default 1e-12).
        dt : float, optional
            Time step in atomic units (default 0.01).
        max_iter : int, optional
            Maximum iterations (default 200).
        chkfile : bool, optional
            Whether to use checkpoint files (default True).
        chkfile_freq : int, optional
            Frequency of checkpoint saves (default 10).
        chkfile_path : str, optional
            Path to checkpoint file (default 'chkfile.txt').
        peak_time_au : float, optional
            Electric field peak time in atomic units (default 1.0).
        width_steps : int, optional
            Electric field width in time steps (default 5).
        shape : str, optional
            Electric field shape ('kick' or 'pulse', default 'kick').
        smoothing : bool, optional
            Whether to smooth the field (default False).
        intensity_au : float, optional
            Field intensity in atomic units (default 5e-5).
        eFieldFile : str, optional
            Electric field output file (default 'eField.csv').
        pFieldFile : str, optional
            Polarization field output file (default 'pField.csv').

        Returns:
        None
        """
        self.pcconv = pcconv
        self.tol_zero = tol_zero
        self.dt = dt
        self.max_iter = max_iter
        self.chkfile = chkfile
        self.chkfile_path = chkfile_path
        self.chkfile_freq = chkfile_freq
        self.peak_time_au = peak_time_au
        self.width_steps = width_steps
        self.shape = shape
        self.smoothing = smoothing
        self.intensity_au = intensity_au
        self.eFieldFile = eFieldFile
        self.pFieldFile = pFieldFile
