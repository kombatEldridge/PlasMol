# chkfile.py
import logging
import numpy as np

logger = logging.getLogger("main")

# Added from TIDES
def restart_from_chkfile(molecule):
    logger.debug(f'Restarting from checkpointed file: {molecule.chkfile}')
    with open(molecule.chkfile, 'r') as f:
        chk_lines = f.readlines()
        molecule.current_time = float(chk_lines[0].split()[3])
        logger.debug(f'Starting at time {molecule.current_time} au.')
        if molecule.nmat == 1:
            molecule.scf.mo_coeff = np.loadtxt(chk_lines[2:], dtype=np.complex128)
        else:
            for idx, line in enumerate(chk_lines):
                if 'Beta' in line:
                    b0 = idx
                    break
            mo_alpha0 = np.loadtxt(chk_lines[3:b0], dtype=np.complex128)
            mo_beta0 = np.loadtxt(chk_lines[b0+1:], dtype=np.complex128)
            molecule.scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))

# Added from TIDES
def update_chkfile(molecule, current_time):
    logger.debug(f'Updating checkpointed file.')
    with open(molecule.chkfile, 'w') as f:
        f.write(f'Current Time (au): {current_time} \nMO Coeffs: \n')
        if molecule.nmat == 1:
            np.savetxt(f, molecule.scf.mo_coeff)
        else:
            f.write('Alpha \n')
            np.savetxt(f, molecule.scf.mo_coeff[0])
            f.write('Beta \n')
            np.savetxt(f, molecule.scf.mo_coeff[1])
