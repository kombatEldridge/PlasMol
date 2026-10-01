# Classical design of a one-pole Drude sphere whose dipole extinction
# sits on the sodium bright root. This is not a PlasMol input.
# nanoparticle.material is still a meep.materials name, so the Medium
# printed at the end cannot be launched.
#
#   /Users/brinton/miniconda/envs/plasmol_env_bea60207/bin/python dielectric.py

from pathlib import Path

import numpy as np
from scipy.special import spherical_jn, spherical_yn

HERE = Path(__file__).resolve().parent
EV_PER_PER_UM = 1.23984193
# Mean of the three LR-TDDFT roots (states 1–3), not 589 nm and not 2.109 eV.
NA_EV = 2.004848723333333
# Campaign JSON value. 1.23984193 / 0.618422 recovers 2.004848 eV.
NA_UM = 0.618422
RADIUS_UM = 0.025
N_HOST = 1.33
# Rakić Au Drude damping (Appl. Opt. 37, 5271, 1998), kept fixed.
# The plasma frequency is the only parameter moved onto the sodium line.
GAMMA_EV = 0.053
EPS_INF = 1.0
SIGMA = 1.0

# meep.materials.Au, energies in eV. Same terms as
# jobs/Au_Na/Models/Gersten-Nitzan/spectrum_model2.py.
_WP2 = 9.03**2
AU_TERMS = [
    ("drude", 0.0, 0.053, 0.760 * _WP2),
    ("lorentz", 0.415, 0.241, 0.024 * _WP2),
    ("lorentz", 0.830, 0.345, 0.010 * _WP2),
    ("lorentz", 2.969, 0.870, 0.071 * _WP2),
    ("lorentz", 4.304, 2.494, 0.601 * _WP2),
    ("lorentz", 13.32, 2.214, 4.384 * _WP2),
]


def drude_eps(energy_ev, omega_p_ev, gamma_ev=GAMMA_EV, eps_inf=EPS_INF, sigma=SIGMA):
    """ε = ε∞ − σ ω_p² / (ω² + i γ ω). Energies in eV. σ = 1."""
    energy = np.asarray(energy_ev, dtype=complex)
    return eps_inf - sigma * omega_p_ev**2 / (energy * (energy + 1j * gamma_ev))


def rakic_au_eps(energy_ev):
    energy = np.asarray(energy_ev, dtype=complex)
    eps = np.ones(np.shape(energy), dtype=complex)
    for kind, e0, gamma, amp in AU_TERMS:
        if kind == "drude":
            eps = eps + amp / (-(energy**2) - 1j * gamma * energy)
        else:
            eps = eps + amp / (e0**2 - energy**2 - 1j * gamma * energy)
    return eps


def _riccati(n, z):
    """ψ, ψ', ξ, ξ' at complex z. ξ uses the spherical Hankel function j + i y."""
    if n < 1:
        raise ValueError("n starts at 1")
    j = spherical_jn(n, z)
    y = spherical_yn(n, z)
    j_down = spherical_jn(n - 1, z)
    y_down = spherical_yn(n - 1, z)
    # ψ' = z j_{n-1} − n j_n
    psi = z * j
    psi_p = z * j_down - n * j
    xi = z * (j + 1j * y)
    xi_p = z * (j_down + 1j * y_down) - n * (j + 1j * y)
    return psi, psi_p, xi, xi_p


def mie_cross_sections(energy_ev, eps_sphere, radius_um=RADIUS_UM, n_host=N_HOST, n_max=2):
    """Absorption and extinction cross sections in μm². Dipole plus quadrupole."""
    energy = np.asarray(energy_ev, dtype=float)
    lam = EV_PER_PER_UM / energy
    k = 2.0 * np.pi * n_host / lam
    x = k * radius_um
    m = np.sqrt(np.asarray(eps_sphere, dtype=complex) / n_host**2)
    c_ext = np.zeros(energy.shape, dtype=float)
    c_abs = np.zeros(energy.shape, dtype=float)
    for n in range(1, n_max + 1):
        psi_x, psi_px, xi_x, xi_px = _riccati(n, x)
        psi_mx, psi_pmx, _, _ = _riccati(n, m * x)
        a = (m * psi_mx * psi_px - psi_x * psi_pmx) / (m * psi_mx * xi_px - xi_x * psi_pmx)
        b = (psi_mx * psi_px - m * psi_x * psi_pmx) / (psi_mx * xi_px - m * xi_x * psi_pmx)
        weight = (2 * n + 1) * 2.0 * np.pi / k**2
        c_ext += weight * np.real(a + b)
        c_abs += weight * (np.real(a + b) - np.abs(a) ** 2 - np.abs(b) ** 2)
    return c_abs, c_ext


def peak_energy(energies, curve):
    i = int(np.argmax(curve))
    if i == 0 or i == len(energies) - 1:
        return float(energies[i]), float(curve[i])
    coeff = np.polyfit(energies[i - 1:i + 2], curve[i - 1:i + 2], 2)
    if coeff[0] == 0.0:
        return float(energies[i]), float(curve[i])
    e = -coeff[1] / (2.0 * coeff[0])
    return float(e), float(np.polyval(coeff, e))


def absorption_peak(eps_of_energy, window):
    grid = np.linspace(window[0], window[1], 4001)
    c_abs, c_ext = mie_cross_sections(grid, eps_of_energy(grid))
    e_abs, _ = peak_energy(grid, c_abs)
    e_ext, _ = peak_energy(grid, c_ext)
    return e_abs, e_ext


def frohlich_omega_p(energy_ev, gamma_ev=GAMMA_EV):
    """ω_p from Re(ε) = −2 ε_host, σ = 1, ε∞ = 1."""
    eps_h = N_HOST**2
    return np.sqrt((EPS_INF + 2.0 * eps_h) * (energy_ev**2 + gamma_ev**2))


def main():
    target = NA_EV
    # Move ω_p until the Mie absorption peak is the sodium root.
    # The Fröhlich value is the start; retardation at 25 nm shifts it.
    omega = frohlich_omega_p(target)

    def peak_at(omega_p):
        e_abs, e_ext = absorption_peak(lambda e: drude_eps(e, omega_p), (target - 0.4, target + 0.4))
        return e_abs, e_ext

    for _ in range(12):
        e_abs, _ = peak_at(omega)
        # ω_p scales the resonance almost linearly this close to Fröhlich.
        omega *= target / e_abs
    e_abs, e_ext = peak_at(omega)
    f_p_um = omega / EV_PER_PER_UM
    gamma_um = GAMMA_EV / EV_PER_PER_UM

    au_abs, au_ext = absorption_peak(rakic_au_eps, (1.6, 3.2))

    lines = [
        "Synthetic Drude sphere, matched to the Na LR-TDDFT bright root.",
        "Not a PlasMol input. meep.materials has no entry for this function.",
        "",
        f"target_eV\t{target:.10f}",
        f"target_um\t{NA_UM:.6f}",
        f"radius_um\t{RADIUS_UM}",
        f"host_index\t{N_HOST}",
        f"eps_inf\t{EPS_INF}",
        f"sigma\t{SIGMA}",
        f"gamma_eV\t{GAMMA_EV}",
        f"gamma_per_um\t{gamma_um:.10f}",
        f"omega_p_eV\t{omega:.10f}",
        f"omega_p_per_um\t{f_p_um:.10f}",
        f"frohlich_omega_p_eV\t{frohlich_omega_p(target):.10f}",
        f"mie_absorption_peak_eV\t{e_abs:.10f}",
        f"mie_extinction_peak_eV\t{e_ext:.10f}",
        f"rakic_Au_mie_absorption_peak_eV\t{au_abs:.10f}",
        f"rakic_Au_mie_extinction_peak_eV\t{au_ext:.10f}",
        f"detuning_Au_minus_Na_eV\t{au_abs - target:.10f}",
        "",
        "Meep object, once a custom medium is accepted:",
        f"mp.Medium(epsilon={EPS_INF}, E_susceptibilities=[mp.DrudeSusceptibility(frequency={f_p_um:.10f}, gamma={gamma_um:.10f}, sigma={SIGMA})])",
    ]
    text = "\n".join(lines) + "\n"
    (HERE / "dielectric.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
