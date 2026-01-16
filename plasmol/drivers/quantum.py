# drivers/quantum.py
import os
import sys
import csv
import copy
import logging
import multiprocessing
import numpy as np

from plasmol.quantum.molecule import MOLECULE
from plasmol.quantum.electric_field import ELECTRICFIELD

from plasmol.quantum.propagators import *
from plasmol.quantum.propagation import propagation
from plasmol.quantum.checkpoint import update_checkpoint

from plasmol.utils.fourier import transform
from plasmol.utils.plotting import show_eField_pField
from plasmol.utils.csv import initCSV, updateCSV, read_field_csv, apply_damping

# Moved to top level
def run_nothing():
    pass


# Moved to top level
def run_computation(params_instance):
    """Execute the core RT-TDDFT computation for a given params object."""
    try:
        logger = logging.getLogger("main")

        time_values = np.arange(0, params_instance.t_end + params_instance.dt, params_instance.dt)
        times = np.linspace(0, time_values[-1], int(len(time_values)))

        molecule = MOLECULE(params_instance)
        field = ELECTRICFIELD(times, params_instance)

        if params_instance.checkpoint_path is not None and os.path.exists(params_instance.checkpoint_path):
            try:
                _ = read_field_csv(params_instance.eField_path)
                _ = read_field_csv(params_instance.pField_path)
                logger.debug(f"Checkpoint file {params_instance.checkpoint_path} found as well as properly formatted field files: {params_instance.eField_path} and {params_instance.pField_path}. Skipping electric/polarizability file generation.")
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
            logging.info(f"{params_instance.dir}-dir: At {current_time} au, the induced dipole is {mu_arr} in au")
            updateCSV(params_instance.pField_path, current_time, *mu_arr)
            if params_instance.checkpoint_path and index % params_instance.checkpoint_freq == 0:
                update_checkpoint(params_instance, molecule, current_time)

    except Exception as err:
        logger.error(f"RT-TDDFT failed: {err}", exc_info=True)
        sys.exit(1)

def run(params):
    """Run the RT-TDDFT computation, either single-process or multi-process based on params.transform.""" 
    if params.transform:
        # Define directions and storage for processes and pField paths
        directions = ['x', 'y', 'z']
        processes = []
        pField_paths = []
        params_copies = []

        # Set up params copies and paths for each direction (always, to support transform)
        for dir in directions:
            dir_path = f"{dir}_dir"
            os.makedirs(dir_path, exist_ok=True)

            # Create a deep copy of params to ensure process safety
            params_copy = copy.deepcopy(params)
            params_copy.dir = dir

            # Update all file paths to use the direction-specific directory
            attr_list = ['eField_path', 'pField_path', 'pField_Transform_path', 'eField_vs_pField_path', 'eV_spectrum_path']
            if params.checkpoint_path is not None:
                attr_list.append('checkpoint_path')
            for attr in attr_list:
                if hasattr(params_copy, attr):
                    original_path = getattr(params_copy, attr)
                    file_name = os.path.basename(original_path)
                    new_path = os.path.join(dir_path, file_name)
                    setattr(params_copy, attr, new_path)

            # Store the pField path for later use in transform
            pField_paths.append(params_copy.pField_path)
            params_copies.append(params_copy)

        # Only spin up and run processes if not do_nothing
        if not params.do_nothing:
            processes = []

            # Create and start a process for each direction
            for params_copy in params_copies:
                process = multiprocessing.Process(target=run_computation, args=(params_copy,))
                processes.append(process)
                process.start()

            # Wait for all processes to complete
            for process in processes:
                process.join()

            # Add damping to the polarizability fields if mu_damping is set
            if params.mu_damping > 0:
                for params_copy in params_copies:
                    mu_arr = read_field_csv(params_copy.pField_path)
                    damped_mu_x, damped_mu_y, damped_mu_z = apply_damping(mu_arr, params.mu_damping)
                    
                    # Write the damped values to a new CSV file
                    damped_path = params_copy.pField_path.replace('.csv', f'_damped.csv')
                    with open(damped_path, 'w', newline='') as file:
                        file.write("# Molecule\'s Polarizability Field intensity in atomic units\n")
                        file.write("\n")
                        writer = csv.writer(file)
                        header = ['Timestamps (au)', 'X Values', 'Y Values', 'Z Values']
                        writer.writerow(header)
                        for t, x, y, z in zip(mu_arr[0], damped_mu_x, damped_mu_y, damped_mu_z):
                            writer.writerow([t, x, y, z])
                    logging.info(f"Damped polarizability field written to {damped_path}")

            # Now plot on the main process for each direction
            for params_copy in params_copies:
                base, _ = os.path.splitext(params.eField_vs_pField_path)
                matplotlibOutput = base + "-" + params_copy.dir
                show_eField_pField(params_copy.eField_path, params_copy.pField_path, matplotlibOutput=matplotlibOutput)
                if params.mu_damping > 0:
                    damped_path = params_copy.pField_path.replace('.csv', f'_damped.csv')
                    damped_output = base + "-" + params_copy.dir + '-damped'
                    show_eField_pField(params_copy.eField_path, damped_path, matplotlibOutput=damped_output)

        # Apply Fourier transform to the pField CSV files
        params_transform = [pField_paths[0], pField_paths[1], pField_paths[2], params.pField_Transform_path, params.eV_spectrum_path]
        if params.fourier_gamma is not None:
            params_transform.append(params.fourier_gamma)
        transform(*params_transform)
    else:
        # Run the original single-process computation
        if not params.do_nothing:
            run_computation(params)
        base, ext = os.path.splitext(params.eField_vs_pField_path)
        show_eField_pField(params.eField_path, params.pField_path, matplotlibOutput=base)