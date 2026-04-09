# main.py
import os
import sys

if __name__ == "__main__" and __package__ is None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    __package__ = os.path.basename(current_dir)

import logging
import numpy as np

from plasmol import constants
from plasmol.drivers import *
from plasmol.utils.logging import setup_logging
from plasmol.utils.input.cli import parse_arguments
from plasmol.utils.input.params import PARAMS

if __name__ == "__main__":
    # Step 1: Grab CLI args
    args = parse_arguments()

    if getattr(args, 'describe', False):
        from plasmol.utils.input.params import PARAMS   # adjust import path as needed
        PARAMS.describe_parameters()
        sys.exit(0)
        
    # Handle existing log file: warn to terminal and use numbered name (hello(1).log etc.)
    # This ensures main, PARAMS, and all children use the *same* log file.
    if args.log and os.path.exists(args.log):
        base, ext = os.path.splitext(args.log)
        counter = 1
        while True:
            candidate = f"{base}({counter}){ext}"
            if not os.path.exists(candidate):
                print(
                    f"WARNING: Log file '{args.log}' already exists. "
                    f"Using '{candidate}' instead.",
                    file=sys.stderr
                )
                args.log = candidate
                break
            counter += 1

    # Set up logging (now using reusable function so child processes can use it too)
    logger = setup_logging(args.verbose, args.log)

    # Step 2: Fill in PARAMS dataclass
    params = PARAMS(args)
    logger.debug(f"PARAMS successfully created: ")
    for key, value in vars(params).items():
        logger.debug(f"\t{key}: {value}")
    
    if args.verbose >= 2:
        with open("run_parameters", "w") as f:
            f.write("=== Run Parameters ===\n\n")
            for key in sorted(vars(params).keys()):
                value = getattr(params, key)
                f.write(f"{key}: {value}\n")

    logger.info(f"The timestep for this simulation is {params.dt} au (roughly {np.round(params.dt / constants.T_AU_FS, decimals=5)} fs).")
    logger.info(f"The simulation will propagate until {params.t_end} au (roughly {np.round(params.t_end / constants.T_AU_FS, decimals=5)} fs).")
    
    if params.has_checkpoint and not params.resumed_from_checkpoint:
        from plasmol.utils.checkpoint import init_checkpoint
        init_checkpoint(params)

    params.fourier_damp = False
    
    # Step 3: Execute proper workflow
    if params.run_molecule_simulation:
        if params.has_comparison:
            run_comparison(params)
        elif params.has_fourier:
            run_fourier(params)
        else:
            run_quantum(params)
    elif params.run_plasmon_simulation:
        run_classical(params)
    else:
        run_plasmol(params)
