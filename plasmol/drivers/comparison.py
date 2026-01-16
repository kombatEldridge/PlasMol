# plasmol/drivers/comparison.py

import logging
import matplotlib.pyplot as plt
import numpy as np

from plasmol.quantum.molecule import MOLECULE

logger = logging.getLogger("main")

def filter_mo_energies(mo_energy, num_virtual=None, y_min=None, y_max=None, min_index=None, max_index=None):
    if mo_energy is None or len(mo_energy) == 0:
        return np.array([]), np.array([])
    
    mo_energy = np.asarray(mo_energy)
    if mo_energy.ndim != 1:
        raise ValueError("mo_energy must be a 1D array")
    
    # Assume mo_energy is already sorted ascending; get original indices
    indices = np.arange(len(mo_energy))
    
    # Apply y_min and y_max filters
    mask = np.ones(len(mo_energy), dtype=bool)
    if y_min is not None:
        mask &= (mo_energy >= y_min)
    if y_max is not None:
        mask &= (mo_energy <= y_max)
    
    # Apply index filters if specified
    if min_index is not None:
        mask &= (indices >= min_index)
    if max_index is not None:
        mask &= (indices <= max_index)
    
    filtered_energies = mo_energy[mask]
    filtered_indices = indices[mask]
    
    # Split into occupied and virtual based on sign
    occ_mask = filtered_energies < 0
    occupied_energies = filtered_energies[occ_mask]
    occupied_indices = filtered_indices[occ_mask]
    virtual_energies = filtered_energies[~occ_mask]
    virtual_indices = filtered_indices[~occ_mask]
    
    # Limit virtual if specified
    if num_virtual is not None:
        virtual_energies = virtual_energies[:num_virtual]
        virtual_indices = virtual_indices[:num_virtual]
    
    # Combine, sort by index to preserve order
    combined_energies = np.concatenate([occupied_energies, virtual_energies])
    combined_indices = np.concatenate([occupied_indices, virtual_indices])
    sort_order = np.argsort(combined_indices)
    
    return combined_energies[sort_order], combined_indices[sort_order]

def get_background_color(mo_energy):
    """
    Determine background color based on MO energies.
    
    - Light transparent red if >5 negative MOs.
    - Light transparent yellow if 6th MO (LUMO) is close to zero (|E| < 1e-3).
    - Light transparent green otherwise.
    
    Parameters:
    mo_energy : np.ndarray
        Full array of sorted MO energies.
    
    Returns:
    str or tuple
        Matplotlib color (with alpha for transparency).
    """
    if mo_energy is None or len(mo_energy) < 6:
        return 'lightgreen'  # Default to green if too few MOs
    
    num_negative = np.sum(mo_energy < 0)
    if num_negative > 5:
        return 'mistyrose'  # Light red
    # if any positive valeus are close to zero
    elif np.any(np.abs(mo_energy[mo_energy >= 0]) < 1e-2):
        return 'lightyellow'
    else:
        return 'lightgreen'

def run(params):
    """
    Run comparisons of MO energies and Gamma matrices across basis sets and XC functionals.
    
    Loops over provided lists of basis sets and XC functionals, computing and logging
    MO energies and (if damping enabled) the Gamma matrix for each combination.
    Generates stem plots for MO energies in separate PNG files for each combination,
    and one large combined plot with subplots for all combinations.
    Allows limiting the number of virtual MOs shown via params.num_virtual,
    and filtering/zooming y-axis range via params.y_min and params.y_max.
    Preserves original MO indices on x-axis.
    Colors plot background based on MO characteristics.
    
    Parameters:
    params : object
        Parameters object with bases, xcs, optional num_virtual, y_min, y_max, and damping settings.
    
    Returns:
    None
    """
    if not hasattr(params, 'bases') or not params.bases:
        logger.error("No basis sets provided for comparison.")
        return
    if not hasattr(params, 'xcs') or not params.xcs:
        logger.error("No XC functionals provided for comparison.")
        return

    # Ensure transform is disabled
    params.transform = False
    
    # Get optional parameters
    num_virtual = getattr(params, 'num_virtual', None)
    if num_virtual is not None:
        logger.info(f"Limiting plots to {num_virtual} virtual MOs above zero energy.")
    y_min = getattr(params, 'y_min', None)
    y_max = getattr(params, 'y_max', None)
    if y_min is not None or y_max is not None:
        logger.info(f"Filtering and zooming y-axis to [{y_min}, {y_max}] hartree.")
    mo_start = getattr(params, 'mo_start', None)
    mo_end = getattr(params, 'mo_end', None)
    if mo_start is not None or mo_end is not None:
        logger.info(f"Filtering plots to MOs {mo_start} through {mo_end} .")
    
    # Collect data for all combinations: dict of (basis, xc) -> (filtered_mo, filtered_indices, full_mo_sorted)
    mo_data = {}
    
    for basis in params.bases:
        for xc in params.xcs:
            logger.info(f"Comparing for basis: {basis}, XC: {xc}")
            params.basis = basis
            params.xc = xc.upper()
            molecule = MOLECULE(params)
            
            logger.info("MO energies (in hartree):")
            logger.info(molecule.mf.mo_energy)
            
            full_mo_sorted = np.sort(molecule.mf.mo_energy)
            filtered_mo, filtered_indices = filter_mo_energies(
                full_mo_sorted, num_virtual, y_min, y_max,
                min_index=(mo_start - 1) if mo_start else None,
                max_index=(mo_end - 1) if mo_end else None
            )
            mo_data[(basis, xc)] = (filtered_mo, filtered_indices, full_mo_sorted)
            
            # Individual plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.stem(filtered_indices + 1, filtered_mo, linefmt='b-', markerfmt='bo', basefmt='r-')
            ax.set_title(f"MO Energies for {basis} / {xc}")
            ax.set_xlabel("MO Index ")
            ax.set_ylabel("Energy (hartree)")
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            ax.set_facecolor(get_background_color(full_mo_sorted))
            ax.grid(True)
            plot_filename = f"img/{basis}_{xc}_mo_energies.png"
            plt.savefig(plot_filename)
            plt.close(fig)
            logger.info(f"MO energies plot saved to: {plot_filename}")
            
            if params.damping is not None:
                gamma = molecule.get_gamma_ao(**molecule.damping_params)
                logger.info("Gamma matrix:")
                logger.info(gamma)
            else:
                logger.info("Damping not enabled; no Gamma matrix computed.")
    
    # Create large combined plot
    num_rows = len(params.bases)
    num_cols = len(params.xcs)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 4), squeeze=False)
    
    for i, basis in enumerate(params.bases):
        for j, xc in enumerate(params.xcs):
            ax = axs[i, j]
            data = mo_data.get((basis, xc))
            if data is not None:
                filtered_mo, filtered_indices, full_mo_sorted = data
                ax.stem(filtered_indices + 1, filtered_mo, linefmt='b-', markerfmt='bo', basefmt='r-')
                ax.set_title(f"{basis} / {xc}", fontsize=18)
                ax.set_xlabel("MO Index ", fontsize=14)
                ax.set_ylabel("Energy (hartree)",   fontsize=14)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
                ax.set_facecolor(get_background_color(full_mo_sorted))
                ax.grid(True)
    
    plt.tight_layout()
    combined_filename = "img/all_mo_energies.png"
    plt.savefig(combined_filename)
    plt.close(fig)
    logger.info(f"Combined MO energies plot saved to: {combined_filename}")