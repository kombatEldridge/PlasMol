# params.py
class PARAMS:
    """
    Container for simulation parameters given from input files and cli inputs.
    """
    def __init__(self, 
                 pcconv=1e-12, 
                 tol_zero=1e-12, 
                 dt=0.01, 
                 max_iter=200, 
                 chkfile=True, 
                 chkfile_freq=10, 
                 chkfile_path='chkfile.npz', 
                 peak_time_au=1.0, 
                 width_steps=5, 
                 shape='kick', 
                 smoothing=False, 
                 intensity_au=5e-5, 
                 eField_path='eField.csv', 
                 pField_path='pField.csv', 
                 pField_Transform_path='pField_Transform.csv',
                eField_vs_pField_path='output.png',
                eV_spectrum_path='spectrum.png',
                 
                 ):

