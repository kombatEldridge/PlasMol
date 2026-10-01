# Synthetic dielectric

The resonant series is a sodium atom and a 25 nm gold sphere. That sphere does not sit on the sodium root. This step is the later check on a dielectric built so that it does.

Same atom, same μ, same radius, same one-pixel gap, same Gaussian. The only change from Steps 3 and 5 is the material inside the sphere. No CAP rows, no Step 6 controls, and no Step 7 gaps. D1 and G2 have no sphere, so they are not repeated. The comparisons are still D3 − D1 and G4 − G2, read against the gold sphere.

The poles do not go in the JSON. `na_drude.py` assigns `meep.materials.NaDrude`, and each nanoparticle block names that file with `material_file` set to `../na_drude.py`. The name has to be new: `load_meep_material` rejects anything that is already a `Medium` in Meep's library, including `Au`. How that load works is `doc/docs/custom_material.md`.

`dielectric.py` is the classical design script, not the file PlasMol loads. It puts one Drude pole on the sodium root for this radius and this water host. ε∞ = 1, σ = 1, and γ = 0.053 eV, which is the free-electron damping in the Rakić gold table and is not refitted. The plasma frequency is the parameter that moves.

| Quantity | Value |
| --- | --- |
| Sodium root | 2.004849 eV, 0.618422 μm |
| Radius, host | 0.025 μm, n = 1.33 |
| γ | 0.053 eV, 0.042747 μm⁻¹ |
| ω_p, quasistatic | 4.272243 eV. This is Re(ε) = −2 ε_host |
| ω_p, used | 4.505664 eV, 3.634064 μm⁻¹. Mie absorption of the 25 nm sphere is the sodium root |
| Extinction peak | 2.005787 eV, about 1 meV above the absorption peak |
| Rakić `Au`, same sphere | Mie absorption 2.340 eV, extinction 2.332 eV. Quasistatic condition Re(ε) = −2 ε_host at 2.420 eV |
| Gold minus sodium | 0.335 eV on the Mie absorption peak |

The quasistatic plasma frequency is not the one to use. At 25 nm the dynamic depolarization shifts the peak, so ω_p is raised until the Mie absorption (dipole and quadrupole) lands on 2.004849 eV. The same check with the quasistatic ω_p and a radius of 0.0005 μm comes back within 1 meV of that root, so the shift is the radius and not the formula. Full figures are in `dielectric.txt`.

`na_drude.py` assigns this object:

```python
mp.Medium(
    epsilon=1.0,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=3.6340635899, gamma=0.0427473847, sigma=1.0)
    ],
)
```

Both frequencies are in μm⁻¹, Meep's `f = ω / 2π` units. The jobs are `D3`, `D4`, `G3`, and `G4`. Each is the Step 3 or Step 5 file with only the nanoparticle block changed, and a copy of `Na.xyz` beside the JSON. Spin is 1. There is no `watch_indices`. The Gaussian wavelength is 0.618422 μm. The molecule x is 0.02645 μm. The plot window on G3 and G4 is 1.5 to 4.5 eV, the same window as Steps 5–7. None of the four files contain a placeholder.

If this sphere moves D3 and G4 and the gold sphere does not, the gold result was the 0.335 eV miss. If both move by about the same amount, the sphere is doing something that does not need its plasmon on the atom. Do not retune μ or ε₀. Do not change the radius to chase the peak.
