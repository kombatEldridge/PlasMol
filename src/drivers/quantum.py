# drivers/quantum.py
import os
import sys
import csv
import copy
import logging
import threading
import numpy as np

from ..quantum.molecule import MOLECULE
from ..quantum.electric_field import ELECTRICFIELD

from ..quantum.propagators import *
from ..quantum.propagation import propagation
from ..quantum.chkfile import update_chkfile

from ..utils.fourier import transform
from ..utils.plotting import show_eField_pField
from ..utils.csv import initCSV, updateCSV, read_field_csv

def run(params):
    """Run the RT-TDDFT computation, either single-threaded or multi-threaded based on params.transform."""
    def run_computation(params_instance):
        """Execute the core RT-TDDFT computation for a given params object."""
        try:
            logger = logging.getLogger("main")

            time_values = np.arange(0, params_instance.t_end + params_instance.dt, params_instance.dt)
            times = np.linspace(0, time_values[-1], int(len(time_values)))

            if params_instance.restart:
                for path in ['eField_path', 'pField_path', 'pField_Transform_path', 'chkfile_path', 'eField_vs_pField_path', 'eV_spectrum_path']:
                    if hasattr(params_instance, path):
                        file_path = getattr(params_instance, path)
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"Deleted {file_path}")
                            except OSError as e:
                                logger.error(f"Error deleting {file_path}: {e}")
                        else:
                            logger.debug(f"No such file: {file_path}")

            molecule = MOLECULE(params_instance)
            field = ELECTRICFIELD(times, params_instance)

            if params_instance.chkfile_path is not None and os.path.exists(params_instance.chkfile_path):
                try:
                    _ = read_field_csv(params_instance.eField_path)
                    _ = read_field_csv(params_instance.pField_path)
                    logger.debug(f"Checkpoint file {params_instance.chkfile_path} found as well as properly formatted field files: {params_instance.eField_path} and {params_instance.pField_path}. Skipping electric/polarizability file generation.")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                initCSV(params_instance.eField_path, "Electric Field intensity in atomic units")
                initCSV(params_instance.pField_path, "Molecule's Polarizability Field intensity in atomic units")
                logger.debug(f"Field files successfully initialized: {params_instance.eField_path} and {params_instance.pField_path}")

            rows = ((t, i0, i1, i2) for t, (i0, i1, i2) in zip(times, field.field))
            with open(params_instance.eField_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            
            logger.debug(f"Electric field successfully added to {params_instance.eField_path}")
        
            # Select propagator
            if params_instance.propagator == "step":
                propagate = propagate_step
            elif params_instance.propagator == "magnus2":
                propagate = propagate_magnus2
            elif params_instance.propagator == "rk4":
                propagate = propagate_rk4

            for index, current_time in enumerate(field.times):
                if current_time < molecule.chkpoint_time:
                    continue
                mu_arr = propagation(params_instance, molecule, field.field[index], propagate)
                logging.info(f"At {current_time} au, combined Bohr output is {mu_arr} in au")
                updateCSV(params_instance.pField_path, current_time, *mu_arr)
                if params_instance.chkfile_path and index % params_instance.chkfile_freq == 0:
                    update_chkfile(params_instance, molecule, current_time)

            # show_eField_pField(params_instance.eField_path, params_instance.pField_path)

        except Exception as err:
            logger.error(f"RT-TDDFT failed: {err}", exc_info=True)
            sys.exit(1)

    if params.transform:
        # Define directions and storage for threads and pField paths
        directions = ['x', 'y', 'z']
        threads = []
        pField_paths = []

        # Set up and start a thread for each direction
        for dir in directions:
            dir_path = f"{dir}_dir"
            os.makedirs(dir_path, exist_ok=True)

            # Create a deep copy of params to ensure thread safety
            params_copy = copy.deepcopy(params)
            params_copy.dir = dir

            # Update all file paths to use the direction-specific directory
            for attr in ['eField_path', 'pField_path', 'pField_Transform_path', 'chkfile_path', 'eField_vs_pField_path', 'eV_spectrum_path']:
                if hasattr(params_copy, attr):
                    original_path = getattr(params_copy, attr)
                    file_name = os.path.basename(original_path)
                    new_path = os.path.join(dir_path, file_name)
                    setattr(params_copy, attr, new_path)

            # Store the pField path for later use in transform
            pField_paths.append(params_copy.pField_path)

            # Create and start a thread for this direction
            thread = threading.Thread(target=run_computation, args=(params_copy,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Apply Fourier transform to the pField CSV files
        transform(pField_paths[0], pField_paths[1], pField_paths[2], params.pField_Transform_path, params.eV_spectrum_path)
    else:
        # Run the original single-threaded computation
        run_computation(params)