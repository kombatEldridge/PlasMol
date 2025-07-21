# constants.py
c = 299792458.0
epsilon_0 = 8.8541878128e-12
angs2bohr = 1.0/0.52917721067

convertTimeMeep2fs = 10 / 3
convertTimeAtomic2fs = 0.024188843
convertMomentAtomic2Meep = 8.4783536198e-30 * c / 1 / 1e-6 / 1e-6
convertTimeMeep2Atomic = convertTimeMeep2fs / convertTimeAtomic2fs
convertFieldMeep2Atomic = 1 / 1e-6 / epsilon_0 / c / 0.51422082e12

A0 = 5.29177210903e-11  # Bohr radius in m
C_AU = 137.035999       # Speed of light in atomic units
D_AU_NM = 1e-9 / A0     # Distance conv in au/nm
T_AU_FS = 41.3413733    # Time conv in au/fs
V_AU_AA3 = 0.14818471    # Volume conv in au/Å³

vdw_radii = {'H': 1.2, 'He': 1.4, 'Li': 2.2, 'Be': 1.9, 'B': 1.8, 'C': 1.7, 'N': 1.6, 'O': 1.55, 'F': 1.5, 
             'Ne': 1.54, 'Na': 2.4, 'Mg': 2.2, 'Al': 2.1, 'Si': 2.1, 'P': 1.95, 'S': 1.8, 'Cl': 1.8, 'Ar': 1.88, 
             'K': 2.8, 'Ca': 2.4, 'Sc': 2.3, 'Ti': 2.15, 'V': 2.05, 'Cr': 2.05, 'Mn': 2.05, 'Fe': 2.05, 'Co': 2.0, 
             'Ni': 2.0, 'Cu': 2.0, 'Zn': 2.1, 'Ga': 2.1, 'Ge': 2.1, 'As': 2.05, 'Se': 1.9, 'Br': 1.9, 'Kr': 2.02, 
             'Rb': 2.9, 'Sr': 2.55, 'Y': 2.4, 'Zr': 2.3, 'Nb': 2.15, 'Mo': 2.1, 'Tc': 2.05, 'Ru': 2.05, 'Rh': 2.0, 
             'Pd': 2.05, 'Ag': 2.1, 'Cd': 2.2, 'In': 2.2, 'Sn': 2.25, 'Sb': 2.2, 'Te': 2.1, 'I': 2.1, 'Xe': 2.16, 
             'Cs': 3.0, 'Ba': 2.7, 'La': 2.5, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45, 'Pm': 2.43, 'Sm': 2.42, 'Eu': 2.4, 
             'Gd': 2.38, 'Tb': 2.37, 'Dy': 2.35, 'Ho': 2.33, 'Er': 2.32, 'Tm': 2.3, 'Yb': 2.28, 'Lu': 2.27, 'Hf': 2.25, 
             'Ta': 2.2, 'W': 2.1, 'Re': 2.05, 'Os': 2.0, 'Ir': 2.0, 'Pt': 2.05, 'Au': 2.1, 'Hg': 2.05, 'Tl': 2.2, 
             'Pb': 2.3, 'Bi': 2.3, 'Po': 2.0, 'At': 2.0, 'Rn': 2.0, 'Fr': 2.0, 'Ra': 2.0, 'Ac': 2.0, 'Th': 2.4, 
             'Pa': 2.0, 'U': 2.3, 'Np': 2.0, 'Pu': 2.0, 'Am': 2.0, 'Cm': 2.0, 'Bk': 2.0, 'Cf': 2.0, 'Es': 2.0, 
             'Fm': 2.0, 'Md': 2.0, 'No': 2.0, 'Lr': 2.0, 'Rf': 2.0, 'Db': 2.0, 'Sg': 2.0, 'Bh': 2.0, 'Hs': 2.0, 
             'Mt': 2.0, 'Ds': 2.0, 'Rg': 2.0, 'Cn': 2.0, 'Nh': 2.0, 'Fl': 2.0, 'Mc': 2.0, 'Lv': 2.0, 'Ts': 2.0, 'Og': 2.0}
