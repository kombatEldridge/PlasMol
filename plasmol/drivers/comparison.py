# plasmol/drivers/comparison.py

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from plasmol.quantum.molecule import MOLECULE

logger = logging.getLogger("main")


def get_background_color(mo_energy_hartree, nocc=None):
    """
    Determine background color based on MO energies (expects Hartree).
    - Mistyrose if more negative MOs than occupied orbitals (negative virtuals)
    - Lightyellow if LUMO close to zero (|E| < 1e-3 hartree)
    - Lightgreen otherwise
    """
    if mo_energy_hartree is None or len(mo_energy_hartree) < 6:
        return 'lightgreen'

    num_negative = np.sum(mo_energy_hartree < 0)
    
    # Generalized for ANY molecule (no longer hard-coded to water's 5 occupied MOs)
    if nocc is not None and num_negative > nocc:
        return 'mistyrose'  # Negative virtual orbitals present
    elif np.any(np.abs(mo_energy_hartree[mo_energy_hartree >= 0]) < 1e-3):
        return 'lightyellow'
    else:
        return 'lightgreen'


def run(params):
    # Ensure output directory exists
    Path(params.comparison_dir_name).mkdir(parents=True, exist_ok=True)
    Path(f"{params.comparison_dir_name}/individuals").mkdir(parents=True, exist_ok=True)

    # Data storage for combined plot:
    # (selected_energies_ev, selected_indices, full_mo_hartree, nocc, homo_ev, lumo_ev)
    mo_data = {}

    for basis in params.comparison_bases:
        for xc in params.comparison_xcs:
            logger.info(f"Comparing for basis: {basis}, XC: {xc}")

            # Set parameters for this combination
            params.molecule_basis = basis
            params.molecule_xc = xc

            molecule = MOLECULE(params)

            # === Data prep ===
            mo_energy_hartree = np.asarray(molecule.mf.mo_energy).copy()
            nmo = len(mo_energy_hartree)
            mo_energy_ev = mo_energy_hartree * 27.21138602

            # HOMO index
            occ_mask = molecule.mf.mo_occ > 0.5
            homo_idx_0 = np.where(occ_mask)[0][-1] if np.any(occ_mask) else 0
            nocc = homo_idx_0 + 1

            # HOMO / LUMO energies (in eV) for annotation
            homo_ev = mo_energy_ev[homo_idx_0]
            lumo_idx = homo_idx_0 + 1
            lumo_ev = mo_energy_ev[lumo_idx] if lumo_idx < nmo else np.nan

            # === Selection logic ===
            if (hasattr(params, 'comparison_index_min') and params.comparison_index_min is not None) or \
               (hasattr(params, 'comparison_index_max') and params.comparison_index_max is not None):
                start_0 = max(0, (params.comparison_index_min or 1) - 1)
                end_0   = min(nmo, (params.comparison_index_max or nmo))
            else:
                n_occ_show = params.comparison_num_occupied if hasattr(params, 'comparison_num_occupied') and params.comparison_num_occupied is not None else nocc
                n_virt_show = params.comparison_num_virtual if hasattr(params, 'comparison_num_virtual') and params.comparison_num_virtual is not None else (nmo - nocc)
                start_0 = max(0, nocc - n_occ_show)
                end_0   = min(nmo, nocc + n_virt_show)

            selected_indices = np.arange(start_0 + 1, end_0 + 1)
            selected_energies_ev = mo_energy_ev[start_0:end_0]

            # Store for combined plot
            mo_data[(basis, xc)] = (selected_energies_ev, selected_indices, mo_energy_hartree, nocc, homo_ev, lumo_ev)

            # === Individual plot ===
            fig, ax = plt.subplots(figsize=(9, 7))

            is_occupied = selected_indices <= nocc
            ax.scatter(selected_indices[is_occupied], selected_energies_ev[is_occupied],
                       color="#1f77b4", s=80, zorder=5, label="Occupied")
            ax.scatter(selected_indices[~is_occupied], selected_energies_ev[~is_occupied],
                       color="#ff7f0e", s=80, zorder=5, label="Virtual")

            for idx, e in zip(selected_indices, selected_energies_ev):
                ax.hlines(e, idx - 0.35, idx + 0.35, color="gray", linewidth=2.2, alpha=0.8)

            ax.set_xlabel("MO (1-Indexed)")
            ax.set_ylabel("MO Energy (eV)")
            ax.set_xticks(selected_indices)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=1)
            ax.set_title(f"MO Energies — {basis} / {xc}")
            ax.legend(loc="upper right")

            # HOMO / LUMO annotation
            gap_ev = lumo_ev - homo_ev if not np.isnan(lumo_ev) else np.nan
            anno_text = (f"HOMO = {homo_ev:.2f} eV\n"
                         f"LUMO = {lumo_ev:.2f} eV\n"
                         f"ΔE   = {gap_ev:.2f} eV")
            ax.text(0.79, 0.13, anno_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
            
            # y_min / y_max (convert Hartree → eV for the plot)
            if (hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None) or \
               (hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None):
                ymin = params.comparison_y_min * 27.21138602 if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None else None
                ymax = params.comparison_y_max * 27.21138602 if hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None else None
                ax.set_ylim(ymin, ymax)

            plt.tight_layout()
            plot_filename = f"{params.comparison_dir_name}/individuals/{basis}_{xc}_mo_energies.png"
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
                selected_ev, selected_idx, full_hartree, this_nocc, homo_ev, lumo_ev = data

                # Same nice style as individual plots
                is_occ = selected_idx <= this_nocc
                ax.scatter(selected_idx[is_occ], selected_ev[is_occ], color="#1f77b4", s=60, label="Occ")
                ax.scatter(selected_idx[~is_occ], selected_ev[~is_occ], color="#ff7f0e", s=60, label="Virt")

                for idx, e in zip(selected_idx, selected_ev):
                    ax.hlines(e, idx - 0.3, idx + 0.3, color="gray", lw=1.8)

                ax.set_title(f"{basis} / {xc}", fontsize=14)
                ax.set_xlabel("MO (1-Indexed)")
                ax.set_ylabel("Energy (eV)")
                ax.grid(True, alpha=0.3)
                ax.set_facecolor(get_background_color(full_hartree, this_nocc))
                ax.axhline(0, color='black', ls='--', alpha=0.4)

                # HOMO / LUMO annotation
                gap_ev = lumo_ev - homo_ev if not np.isnan(lumo_ev) else np.nan
                anno_text = (f"HOMO = {homo_ev:.2f} eV\n"
                             f"LUMO = {lumo_ev:.2f} eV\n"
                             f"ΔE   = {gap_ev:.2f} eV")
                ax.text(0.7, 0.2, anno_text, transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

                # y_min / y_max (convert Hartree → eV)
                if (hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None) or \
                   (hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None):
                    ymin = params.comparison_y_min * 27.21138602 if hasattr(params, 'comparison_y_min') and params.comparison_y_min is not None else None
                    ymax = params.comparison_y_max * 27.21138602 if hasattr(params, 'comparison_y_max') and params.comparison_y_max is not None else None
                    ax.set_ylim(ymin, ymax)

    # Subtitle at the BOTTOM in italics as one continuous paragraph
    fig.suptitle("Comparison of Molecular Orbital Energies", fontsize=22, y=1.02)

    subtitle_text = (
        "Background color scheme: Red when more MOs have negative energy than the number "
        "of occupied orbitals (indicating negative virtual orbitals), yellow when the LUMO "
        "energy is very close to zero ($> 10^{-3}$ Ha), otherwise green."
    )
    fig.text(0.5, 0.02, subtitle_text, ha='center', va='bottom', fontsize=12, transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0.07, 1, 1])   # Leave space at bottom for subtitle
    combined_filename = f"{params.comparison_dir_name}/all_mo_energies.png"
    fig.savefig(combined_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined MO energies plot saved to: {combined_filename}")