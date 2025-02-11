import numpy as np
import os
import inspect


mapInt2Char = {0: 'x', 1: 'y', 2: 'z'}


def get_U_t():
    if os.path.exists('U_t.npy'):
        U = np.load('U_t.npy')
    else:
        U = None
    return U


def set_U_t(U_t):
    if U_t is None:
        pass
    else:
        np.save('U_t.npy', U_t)


def get_D_mo_0():
    if os.path.exists('D_mo_0.npy'):
        D_mo_0 = np.load('D_mo_0.npy')
    else:
        raise FileNotFoundError(f"D_mo_0.npy has not been initialized yet.")
    return D_mo_0


def set_D_mo_0(D_mo_0):
    np.save('D_mo_0.npy', D_mo_0)


def get_F_mo_0():
    if os.path.exists('F_mo_0.npy'):
        F_mo_0 = np.load('F_mo_0.npy')
    else:
        raise FileNotFoundError(f"F_mo_0.npy has not been initialized yet.")
    return F_mo_0


def set_F_mo_0(F_mo_0):
    np.save('F_mo_0.npy', F_mo_0)


def get_F_mo_t():
    if os.path.exists('F_mo_t.npy'):
        F_mo_t = np.load('F_mo_t.npy')
    else:
        F_mo_t = get_F_mo_0()
    return F_mo_t


def set_F_mo_t(F_mo_t):
    np.save('F_mo_t.npy', F_mo_t)


def get_F_ao_t():
    if os.path.exists('F_ao_t.npy'):
        F_ao_t = np.load('F_ao_t.npy')
    else:
        F_ao_t = get_F_ao_0() #hmmmm
    return F_ao_t


def set_F_ao_t(F_ao_t):
    np.save('F_ao_t.npy', F_ao_t)


def get_F_mo_t_plus_half_dt():
    if os.path.exists('F_mo_t_plus_half_dt.npy'):
        F_mo_t_plus_half_dt = np.load('F_mo_t_plus_half_dt.npy')
    else:
        F_mo_t_plus_half_dt = get_F_mo_0()
    return F_mo_t_plus_half_dt


def set_F_mo_t_plus_half_dt(F_mo_t_plus_half_dt):
    np.save('F_mo_t_plus_half_dt.npy', F_mo_t_plus_half_dt)


def get_F_mo_t_minus_half_dt():
    if os.path.exists('F_mo_t_minus_half_dt.npy'):
        F_mo_t_minus_half_dt = np.load('F_mo_t_minus_half_dt.npy')
    else:
        F_mo_t_minus_half_dt = get_F_mo_0()
    return F_mo_t_minus_half_dt


def set_F_mo_t_minus_half_dt(F_mo_t_minus_half_dt):
    np.save('F_mo_t_minus_half_dt.npy', F_mo_t_minus_half_dt)


def get_D_mo_t_plus_half_dt():
    if os.path.exists('D_mo_t_plus_half_dt.npy'):
        D_mo_t_plus_half_dt = np.load('D_mo_t_plus_half_dt.npy')
    else:
        D_mo_t_plus_half_dt = get_D_mo_0()
    return D_mo_t_plus_half_dt


def set_D_mo_t_plus_half_dt(D_mo_t_plus_half_dt):
    np.save('D_mo_t_plus_half_dt.npy', D_mo_t_plus_half_dt)


def get_U_t_plus_half_dt():
    if os.path.exists('U_t_plus_half_dt.npy'):
        U_t_plus_half_dt = np.load('U_t_plus_half_dt.npy')
    else:
        U_t_plus_half_dt = None
    return U_t_plus_half_dt


def set_U_t_plus_half_dt(U_t_plus_half_dt):
    if U_t_plus_half_dt is None:
        pass
    else:
        np.save('U_t_plus_half_dt.npy', U_t_plus_half_dt)


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