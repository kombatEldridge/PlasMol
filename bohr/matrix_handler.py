import numpy as np
import os

mapInt2Char = {0: 'x', 1: 'y', 2: 'z'}


def get_U_ct():
    if os.path.exists('U_ct.npy'):
        U = np.load('U_ct.npy')
    else:
        U = None
    return U


def set_U_ct(U_ct):
    np.save('U_ct.npy', U_ct)


def get_D_mo_0():
    if os.path.exists('D_mo_0.npy'):
        D_mo_0 = np.load('D_mo_0.npy')
    else:
        raise FileNotFoundError(f"D_mo_0.npy has not been inistialized yet.")
    return D_mo_0


def set_D_mo_0(D_mo_0):
    np.save('D_mo_0.npy', D_mo_0)


def get_F_mo_0():
    if os.path.exists('F_mo_0.npy'):
        F_mo_0 = np.load('F_mo_0.npy')
    else:
        raise FileNotFoundError(f"F_mo_0.npy has not been inistialized yet.")
    return F_mo_0


def set_F_mo_0(F_mo_0):
    np.save('F_mo_0.npy', F_mo_0)


def get_F_mo_ct():
    if os.path.exists('F_mo_ct.npy'):
        F_mo_ct = np.load('F_mo_ct.npy')
    else:
        F_mo_ct = get_F_mo_0()
    return F_mo_ct


def set_F_mo_ct(F_mo_ct):
    np.save('F_mo_ct.npy', F_mo_ct)


def get_F_mo_ndt2():
    if os.path.exists('F_mo_ndt2.npy'):
        F_mo_ndt2 = np.load('F_mo_ndt2.npy')
    else:
        F_mo_ndt2 = get_F_mo_0()
    return F_mo_ndt2


def set_F_mo_ndt2(F_mo_ndt2):
    np.save('F_mo_ndt2.npy', F_mo_ndt2)


def get_D_mo_ct():
    if os.path.exists('D_mo_ct.npy'):
        D_mo_ct = np.load('D_mo_ct.npy')
    else:
        D_mo_ct = get_D_mo_0()
    return D_mo_ct


def set_D_mo_ct(D_mo_ct):
    np.save('D_mo_ct.npy', D_mo_ct)


def clear_Matrix_Files():
    for dir in ['x', 'y', 'z']:
        for file in ['U_ct.npy',
                     'D_mo_0.npy',
                     'D_mo_ct.npy',
                     'F_mo_0.npy',
                     'F_mo_ct.npy',
                     'F_mo_ndt2.npy']:
            try:
                os.remove(file)
            except:
                pass
