# options.py
class OPTIONS:
    def __init__(self):
        # molecule geometry + charge/spin
        self.molecule = {}
        self.charge   = 0
        self.spin     = 0
        self.units    = "bohr"

        # basis and SCF
        self.basis            = None
        self.diis             = True
        self.e_conv           = 1e-6
        self.d_conv           = 1e-6
        self.maxiter          = 200

        # TDDFT / propagation
        self.nroots           = 5
        self.xc               = "lda"
        self.resplimit        = 1e-20
        self.guess_mos        = None
        self.propagator       = "magnus2"
        self.method           = "rttddft"