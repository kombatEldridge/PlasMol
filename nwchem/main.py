import os
import sys
import logging
import numpy as np

from field import FIELD
from molecule import MOLECULE
from logging_utils import PrintLogger
from electric_field_generator import ElectricFieldGenerator
from csv_utils import initCSV, updateCSV
from plotting import show_eField_pField
from cli import parse_arguments
from volume import get_volume
from magnus2 import prop_magnus_ord2_interpol, transform_D_mo_to_D_ao

# main.py
if __name__ == "__main__":
    try:
        # Set up logging
        log_format = '%(levelname)s: %(message)s'
        args = parse_arguments()
        logger = logging.getLogger()
        # Clear any pre-existing handlers.
        if logger.hasHandlers():
            logger.handlers.clear()

        # Set log level based on verbosity.
        if args.verbose >= 2:
            logger.setLevel(logging.DEBUG)
        elif args.verbose == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Use FileHandler if a log file is specified; otherwise, use StreamHandler.
        if args.log:
            handler = logging.FileHandler(args.log, mode='w')
        else:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(handler)

        # Optional: Prevent propagation to avoid duplicate logging
        logger.propagate = False
    
        sys.stdout = PrintLogger(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        logger.debug(f"Arguments given and parsed correctly: {args}")
    
        # Initialize the molecule and extract parameters
        molecule = MOLECULE(args.bohr, args.pcconv)
    
        # Parameters for the electric field
        wavelength = 250 # In nanometers
        peak_time = 25   # time of pulse peak in fs
        width = 0.05     # width parameter in fs^-2
        t_start = 0      # start time in fs
        t_end = 50.0     # end time in fs
        dt_fs = args.dt     # time step in fs

        # Create time array
        time_values = np.arange(t_start, t_end + dt_fs, dt_fs)

        # Initialize the electric field generator
        field_generator = ElectricFieldGenerator(wavelength, peak_time, width)

        # Create an interpolator for the electric field components
        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
    
        # Generate a new time grid with adjusted resolution (args.mult is a multiplier)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        time_step_fs = interpolated_times[1] - interpolated_times[0]
        interpolated_fields = field_generator.get_field_at(interpolated_times, args.dir)

        # Convert time in fs to au
        dt = dt_fs * 41.3413733
        logger.info(f"The timestep for this simulation is {dt} in au or {dt_fs} in fs.")

        # Initialize the interpolated electric field CSV file using initCSV
        interpolated_e_field_csv = "interpolated-E-Field.csv"
        initCSV(interpolated_e_field_csv, "Interpolated Electric Field measured in au.")
    
        # Append the interpolated data rows to the CSV file
        with open(interpolated_e_field_csv, 'a', newline='') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            for t, field in zip(interpolated_times, interpolated_fields):
                writer.writerow([t, field[0], field[1], field[2]])
    
        class Params:
            tol_zero = args.pcconv # previously tol_interpol
            checklvl = 2 # ensure hermitian and unitary
            exp_method = 1  # scipy.expm()
            dt = dt_fs # time difference
            terms_interpol = 2
            lskip_interpol = False
            molecule = MOLECULE(args.bohr, args.pcconv)
            field = FIELD()
            nmats = 1

        params = Params()

        # Initialize CSV file for the polarizability field output
        polarizability_csv = "magnus-P-Field.csv"
        initCSV(polarizability_csv, "Molecule's Polarizability Field measured in au.")
    
        # Log non-comment lines from the Bohr input file
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))
        
        def JK(wfn, D_ao):
            """
            Computes the effective potential and returns the Fock matrix component.

            Parameters:
                wfn: Wavefunction object containing molecular integrals.
                D_ao: Density matrix in atomic orbital basis.

            Returns:
                np.ndarray: The computed Fock matrix.
            """
            pot = wfn.jk.get_veff(wfn.ints_factory, 2 * D_ao)
            Fa = wfn.T + wfn.Vne + pot
            return Fa

        def tdfock(params, D_ao):
            """
            Builds the Fock matrix by including the external field.

            Parameters:
                wfn: Wavefunction object.
                D_ao (np.ndarray): Density matrix in atomic orbital basis.
                exc (array-like): External electric field components.

            Returns:
                np.ndarray: The computed Fock matrix.
            """
            wfn = params.molecule.wfn
            exc = params.field.get_exc_t_plus_dt()
            
            ext = sum(wfn.mu[dir] * exc[dir] for dir in range(3))
            F_ao = JK(wfn, D_ao) - ext
            return F_ao
            
        for index, current_time in enumerate(interpolated_times):
            if index >= 2: params.field.set_exc_t_minus_2dt(interpolated_fields[index-2])
            if index >= 1: params.field.set_exc_t_minus_dt(interpolated_fields[index-1])
            
            F_mo_n12dt = params.molecule.get_F_mo_t_minus_half_dt()
            F_mo = params.molecule.get_F_mo_t()
            D_mo = params.molecule.get_D_mo_t()
            params.field.set_exc_t(interpolated_fields[index])
            params.field.set_exc_t_plus_dt(interpolated_fields[index + 1] if index + 1 < len(interpolated_fields) else interpolated_fields[index])
            
            # mu.py
            mu_arr = np.zeros(3)
            D_mo_t_plus_dt, params = prop_magnus_ord2_interpol(params, tdfock, F_mo_n12dt, F_mo, D_mo)
            D_ao_t_plus_dt = transform_D_mo_to_D_ao(D_mo_t_plus_dt, params)
            volume = get_volume(molecule.molecule["coords"])

            for i in [0, 1, 2]:
                mu_arr[i] = 2 * float((np.trace(params.molecule.wfn.mu[i] @ D_ao_t_plus_dt) - np.trace(params.molecule.wfn.mu[i] @ params.molecule.wfn.D[0])).real)

            mu_arr /= volume

            logging.debug(f"At {current_time} fs, combined Bohr output is {mu_arr} in au")
            updateCSV(polarizability_csv, current_time, *mu_arr)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        logging.info("Simulation completed successfully.")
    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")

