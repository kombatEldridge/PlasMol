import numpy as np
import os
import inspect


mapInt2Char = {0: 'x', 1: 'y', 2: 'z'}


def get_U_ct():
    if os.path.exists('U_ct.npy'):
        U = np.load('U_ct.npy')
    else:
        U = None
    return U


def set_U_ct(U_ct):
    if U_ct is None:
        pass
    else:
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


def get_F_mo_dt2():
    if os.path.exists('F_mo_dt2.npy'):
        F_mo_dt2 = np.load('F_mo_dt2.npy')
    else:
        F_mo_dt2 = get_F_mo_0()
    return F_mo_dt2


def set_F_mo_dt2(F_mo_dt2):
    np.save('F_mo_dt2.npy', F_mo_dt2)


def get_D_mo_dt2():
    if os.path.exists('D_mo_dt2.npy'):
        D_mo_dt2 = np.load('D_mo_dt2.npy')
    else:
        D_mo_dt2 = get_D_mo_0()
    return D_mo_dt2


def set_D_mo_dt2(D_mo_dt2):
    np.save('D_mo_dt2.npy', D_mo_dt2)


def get_U_dt2():
    if os.path.exists('U_dt2.npy'):
        U_dt2 = np.load('U_dt2.npy')
    else:
        U_dt2 = None
    return U_dt2


def set_U_dt2(U_dt2):
    if U_dt2 is None:
        pass
    else:
        np.save('U_dt2.npy', U_dt2)


def clear_Matrix_Files():
    # Get the current module's functions
    functions = inspect.getmembers(inspect.getmodule(clear_Matrix_Files), inspect.isfunction)
    
    # Identify all 'get_' functions
    file_names = []
    for name, func in functions:
        if name.startswith('get_'):
            # Extract the '.npy' file names used in the function
            source = inspect.getsource(func)
            for line in source.splitlines():
                if "os.path.exists(" in line:
                    start = line.find("'") + 1
                    end = line.find(".npy'") + 4
                    file_names.append(line[start:end])
    
    # Remove duplicates and delete files
    for file in set(file_names):
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except FileNotFoundError:
            pass