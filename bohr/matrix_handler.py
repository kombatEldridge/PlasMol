import numpy as np
import os
import inspect


int_to_char ={0: 'x', 1: 'y', 2: 'z'}


def get_U_t(dir):
    if os.path.exists(f'U_t_{int_to_char[dir]}.npy'):
        U = np.load(f'U_t_{int_to_char[dir]}.npy')
    else:
        U = None
    return U


def set_U_t(U_t, dir):
    if U_t is None:
        pass
    else:
        np.save(f'U_t_{int_to_char[dir]}.npy', U_t)


def get_D_mo_0():
    if os.path.exists(f'D_mo_0.npy'):
        D_mo_0 = np.load(f'D_mo_0.npy')
    else:
        raise FileNotFoundError(f"D_mo_0.npy has not been initialized yet.")
    return D_mo_0


def set_D_mo_0(D_mo_0):
    np.save(f'D_mo_0.npy', D_mo_0)


def get_F_mo_0():
    if os.path.exists(f'F_mo_0.npy'):
        F_mo_0 = np.load(f'F_mo_0.npy')
    else:
        raise FileNotFoundError(f"F_mo_0.npy has not been initialized yet.")
    return F_mo_0


def set_F_mo_0(F_mo_0):
    np.save(f'F_mo_0.npy', F_mo_0)


def get_F_mo_t(dir):
    if os.path.exists(f'F_mo_t_{int_to_char[dir]}.npy'):
        F_mo_t = np.load(f'F_mo_t_{int_to_char[dir]}.npy')
    else:
        F_mo_t = get_F_mo_0()
    return F_mo_t


def set_F_mo_t(F_mo_t, dir):
    np.save(f'F_mo_t_{int_to_char[dir]}.npy', F_mo_t)


def get_F_mo_t_minus_half_dt(dir):
    if os.path.exists(f'F_mo_t_minus_half_dt_{int_to_char[dir]}.npy'):
        F_mo_t_minus_half_dt = np.load(f'F_mo_t_minus_half_dt_{int_to_char[dir]}.npy')
    else:
        F_mo_t_minus_half_dt = get_F_mo_0()
    return F_mo_t_minus_half_dt


def set_F_mo_t_minus_half_dt(F_mo_t_minus_half_dt, dir):
    np.save(f'F_mo_t_minus_half_dt_{int_to_char[dir]}.npy', F_mo_t_minus_half_dt)


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