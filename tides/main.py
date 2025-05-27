import os
import sys
import logging
import numpy as np

from params import PARAMS
from molecule import MOLECULE
from logging_utils import PRINTLOGGER
from electric_field import ELECTRICFIELD

from fourier import fourier
from cli import parse_arguments
from propagation import propagation
from plotting import show_eField_pField
from csv_utils import initCSV, updateCSV

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
    
        sys.stdout = PRINTLOGGER(logger, logging.INFO)
        logging.getLogger("h5py").setLevel(logging.INFO)
    
        T_AU_FS = 41.3413733  # Time units in au/fs

        # Convert time in fs to au
        if (args.time_units == 'fs'):
            dt_au = args.dt * T_AU_FS
            dt_fs = args.dt
            t_end = args.t_end * T_AU_FS
            t_end_fs = args.t_end
        elif (args.time_units == 'au'):
            dt_au = args.dt
            dt_fs = args.dt / T_AU_FS
            t_end = args.t_end
            t_end_fs = args.t_end / T_AU_FS
        else: 
            raise ValueError(f"The timestep unit for this simulation should only either be 'fs' or 'au'.")

        # Time step check
        if (dt_au > 0.1):
            raise ValueError(f"The timestep for this simulation are too large to elicit physical results.")

        time_values = np.arange(0, t_end + dt_au, dt_au)
        interpolated_times = np.linspace(0, time_values[-1], int(len(time_values) * args.mult))
        logger.info(f"The timestep for this simulation is {dt_au} in au or {dt_fs} in fs.")
        logger.info(f"The simulation will propagate until {t_end} in au or {t_end_fs} in fs.")
        logger.debug(f"There will be {len(interpolated_times)} timesteps until the simulation finishes.")
        logger.debug(f"Arguments given and parsed successfully: {args}")

        params = PARAMS(
            pcconv=args.pcconv, 
            tol_zero=args.tol_zero, 
            doublecheck=args.doublecheck, 
            exp_method=args.exp_method, 
            dt=dt_au, 
            terms_interpol=args.terms_interpol,
            max_iter=args.max_iter, 
            chkfile=args.chkfile,
            chkfile_path=args.chkfile_path,
            chkfile_freq=args.chkfile_freq,
            peak_time_au=args.peak_time_au,
            width_steps=args.width_steps,
            shape=args.shape,
            smoothing=args.smoothing,
            intensity_au=args.intensity_au,
            eFieldFile=args.eFieldFile,
            pFieldFile=args.pFieldFile,
            )
        
        molecule = MOLECULE(args.bohr, params)

        logger.debug("Building an interpolation profile with ElectricFieldInterpolator")
        field = ELECTRICFIELD(
            interpolated_times, 
            args.dir,
            peak_time_au=params.peak_time_au,
            width_steps=params.width_steps,
            dt=dt_au,
            shape=params.shape,
            smoothing=params.smoothing,
            intensity_au=params.intensity_au,
        )

        if params.chkfile is not None and os.path.exists(params.chkfile_path):
            # assume the eField and pField files have already been built 
            # and you do not need to re-initialize them
            print("ere")
            logger.debug(f"Checkpoint file {params.chkfile_path} found. Skipping electric/polarizability field generation.")
            interpolated_e_field_csv = params.eFieldFile
            polarizability_csv = params.pFieldFile
        else:            
            interpolated_e_field_csv = "eField.csv"  # TODO: add to input file
            initCSV(interpolated_e_field_csv, "Electric Field intensity in atomic units")
            
            # Append the interpolated data rows to the CSV file
            for t, intensity in zip(interpolated_times, field.field):
                updateCSV(interpolated_e_field_csv, t, intensity[0], intensity[1], intensity[2])
            
            logger.debug(f"Electric field successfully built and saved to {interpolated_e_field_csv}")

            # Initialize CSV file for the polarizability field output
            polarizability_csv = "pField.csv"  # TODO: add to input file
            initCSV(polarizability_csv, "Molecule's Polarizability Field intensity in atomic units")

        # Log non-comment lines from the Bohr input file
        logger.info("Bohr input file processed:")
        bohr_input_path = os.path.abspath(args.bohr)
        with open(bohr_input_path, 'r') as bohr_file:
            for line in bohr_file:
                if not line.strip().startswith(('#', '--', '%')):
                    logger.info('\t%s', line.rstrip('\n'))

        logger.debug(f"Moleucle's response will be saved to {polarizability_csv}")
    
        # Main code call
        propagation(params, molecule, field, polarizability_csv)

        # Plot the results using the interpolated electric field data
        show_eField_pField(interpolated_e_field_csv, polarizability_csv)
        fourier(polarizability_csv)
        logging.info("Simulation completed successfully.")

    except Exception as err:
        logging.exception(f"Simulation aborted due to an error: {err}")
