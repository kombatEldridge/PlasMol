# plasmol/drivers/comparison.py

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plasmol.quantum.molecule import MOLECULE

logger = logging.getLogger("main")


def get_background_color(mo_energy_hartree):
    """
    Determine background color based on MO energies (expects Hartree).
    - Light transparent red if >5 negative MOs
    - Light transparent yellow if LUMO close to zero (|E| < 1e-3 hartree)
    - Light transparent green otherwise
    """
    if mo_energy_hartree is None or len(mo_energy_hartree) < 6:
        return 'lightgreen'

    num_negative = np.sum(mo_energy_hartree < 0)
    if num_negative > 5:
        return 'mistyrose'  # Light red
    elif np.any(np.abs(mo_energy_hartree[mo_energy_hartree >= 0]) < 1e-3):
        return 'lightyellow'
    else:
        return 'lightgreen'


def run(params):
    # Ensure output directory exists
    Path("img").mkdir(parents=True, exist_ok=True)

    # Data storage for combined plot: (selected_energies_ev, selected_indices, full_mo_hartree, nocc)
    mo_data = {}

    for basis in params.comparison_bases:
        for xc in params.comparison_xcs:
            logger.info(f"Comparing for basis: {basis}, XC: {xc}")

            # Set parameters for this combination
            params.molecule_basis = basis
            params.molecule_xc = xc

            molecule = MOLECULE(params)

            # === Data prep (fixed — no sorting!) ===
            mo_energy_hartree = np.asarray(molecule.mf.mo_energy).copy()
            nmo = len(mo_energy_hartree)
            mo_energy_ev = mo_energy_hartree * 27.21138602

            # HOMO index
            occ_mask = molecule.mf.mo_occ > 0.5
            homo_idx_0 = np.where(occ_mask)[0][-1] if np.any(occ_mask) else 0
            nocc = homo_idx_0 + 1

            # === Selection logic (exactly as you wanted) ===
            if hasattr(params, 'comparison_index_min') and params.comparison_index_min is not None or hasattr(params, 'comparison_index_max') and params.comparison_index_max is not None:
                start_0 = max(0, (params.comparison_index_min or 1) - 1)
                end_0   = min(nmo, (params.comparison_index_max or nmo))
            else:
                if hasattr(params, 'comparison_num_occupied') and params.comparison_num_occupied is not None:
                    n_occ_show = params.comparison_num_occupied
                else:
                    n_occ_show = nocc

                if hasattr(params, 'comparison_num_virtual') and params.comparison_num_virtual is not None:
                    n_virt_show = params.comparison_num_virtual
                else:
                    n_virt_show = nmo - nocc

                start_0 = max(0, nocc - n_occ_show)
                end_0   = min(nmo, nocc + n_virt_show)

            selected_indices = np.arange(start_0 + 1, end_0 + 1)
            selected_energies_ev = mo_energy_ev[start_0:end_0]

            # Store for combined plot
            mo_data[(basis, xc)] = (selected_energies_ev, selected_indices, mo_energy_hartree, nocc)

            # === Individual plot (publication style) ===
            fig, ax = plt.subplots(figsize=(9, 7))

            is_occupied = selected_indices <= nocc
            ax.scatter(selected_indices[is_occupied], selected_energies_ev[is_occupied],
                       color="#1f77b4", s=80, zorder=5, label="Occupied")
            ax.scatter(selected_indices[~is_occupied], selected_energies_ev[~is_occupied],
                       color="#ff7f0e", s=80, zorder=5, label="Virtual")

            for idx, e in zip(selected_indices, selected_energies_ev):
                ax.hlines(e, idx - 0.35, idx + 0.35, color="gray", linewidth=2.2, alpha=0.8)

            ax.set_xlabel("Molecular Orbital Index")
            ax.set_ylabel("MO Energy (eV)")
            ax.set_xticks(selected_indices)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)
            ax.set_title(f"MO Energies — {basis} / {xc}")
            ax.legend(loc="upper right")

            # y_min/y_max view-only clipping
            if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None or hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None:
                ymin = params.comparison_y_min if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None else None
                ymax = params.comparison_y_max if hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None else None
                ax.set_ylim(ymin, ymax)

            plt.tight_layout()
            plot_filename = f"img/{basis}_{xc}_mo_energies.png"
            fig.savefig(plot_filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"MO diagram saved: {plot_filename}")

    # === Combined grid plot ===
    num_rows = len(params.comparison_bases)
    num_cols = len(params.comparison_xcs)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 5), squeeze=False)

    for i, basis in enumerate(params.comparison_bases):
        for j, xc in enumerate(params.comparison_xcs):
            ax = axs[i, j]
            data = mo_data.get((basis, xc))
            if data:
                selected_ev, selected_idx, full_hartree, this_nocc = data

                # Same nice style as individual plots
                is_occ = selected_idx <= this_nocc
                ax.scatter(selected_idx[is_occ], selected_ev[is_occ], color="#1f77b4", s=60, label="Occ")
                ax.scatter(selected_idx[~is_occ], selected_ev[~is_occ], color="#ff7f0e", s=60, label="Virt")

                for idx, e in zip(selected_idx, selected_ev):
                    ax.hlines(e, idx - 0.3, idx + 0.3, color="gray", lw=1.8)

                ax.set_title(f"{basis} / {xc}", fontsize=14)
                ax.set_xlabel("MO Index")
                ax.set_ylabel("Energy (eV)")
                ax.grid(True, alpha=0.3)
                ax.set_facecolor(get_background_color(full_hartree))
                ax.axhline(0, color='black', ls='--', alpha=0.4)
                if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None or hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None:
                    ymin = params.comparison_y_min if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None else None
                    ymax = params.comparison_y_max if hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None else None
                    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    combined_filename = "img/all_mo_energies.png"
    fig.savefig(combined_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined MO energies plot saved to: {combined_filename}")