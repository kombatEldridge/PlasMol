# chkfile.py
import logging
import numpy as np

logger = logging.getLogger("main")

# Added from TIDES
def restart_from_chkfile(wfn):
    logger.debug(f'Restarting from checkpointed file: {wfn.chkfile}')
    with open(wfn.chkfile, 'r') as f:
        chk_lines = f.readlines()
        current_time = float(chk_lines[0].split()[3])
        logger.debug(f'Starting at time {wfn.current_time} au.')
        if wfn.nmat == 1:
            wfn._scf.mo_coeff = np.loadtxt(chk_lines[2:], dtype=np.complex128)
        else:
            for idx, line in enumerate(chk_lines):
                if 'Beta' in line:
                    b0 = idx
                    break
            mo_alpha0 = np.loadtxt(chk_lines[3:b0], dtype=np.complex128)
            mo_beta0 = np.loadtxt(chk_lines[b0+1:], dtype=np.complex128)
            wfn._scf.mo_coeff = np.stack((mo_alpha0, mo_beta0))
    return current_time

# Added from TIDES
def update_chkfile(wfn, current_time):
    logger.debug(f'Updating checkpointed file.')
    with open(wfn.chkfile, 'w') as f:
        f.write(f'Current Time (au): {current_time} \nMO Coeffs: \n')
        if wfn.nmat == 1:
            np.savetxt(f, wfn._scf.mo_coeff)
        else:
            f.write('Alpha \n')
            np.savetxt(f, wfn._scf.mo_coeff[0])
            f.write('Beta \n')
            np.savetxt(f, wfn._scf.mo_coeff[1])
