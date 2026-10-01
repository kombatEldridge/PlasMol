# Step 9 sphere. dielectric.py designed this pole; this file assigns it.
# Frequencies are Meep um^-1 (f = omega / 2 pi). Not a built-in material.
import meep as mp
import meep.materials as materials

materials.NaDrude = mp.Medium(
    epsilon=1.0,
    E_susceptibilities=[
        mp.DrudeSusceptibility(frequency=3.6340635899, gamma=0.0427473847, sigma=1.0)
    ],
)
