import numpy as np
import meep as mp
from meep.materials import Au_JC_visible as Au

r = 0.025

wvl_min = 0.400 
wvl_max = 0.800 
frq_min = 1/wvl_max
frq_max = 1/wvl_min
frq_cen = 0.5*(frq_min+frq_max)
dfrq = frq_max-frq_min
nfrq = 50

resolution = 1000

dpml = 0.01

pml_layers = [mp.PML(thickness=dpml)]

symmetries = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

s = 0.1
cell_size = mp.Vector3(s, s, s)

half_frame = s*resolution/2

sources = [
    mp.Source(
        mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
        center=mp.Vector3(-0.5*s+dpml),
        size=mp.Vector3(0, s, s),
        component=mp.Ez,
    )
]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
    default_material=mp.Medium(index=1.33)
)

box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

sim.run(until_after_sources=10)

freqs = mp.get_flux_freqs(box_x1)
box_x1_data = sim.get_flux_data(box_x1)
box_x2_data = sim.get_flux_data(box_x2)
box_y1_data = sim.get_flux_data(box_y1)
box_y2_data = sim.get_flux_data(box_y2)
box_z1_data = sim.get_flux_data(box_z1)
box_z2_data = sim.get_flux_data(box_z2)
box_x1_flux0 = mp.get_fluxes(box_x1)

sim.reset_meep()

# ------- with sphere (SCAT) ---------

geometry = [mp.Sphere(radius=r,
                      center=mp.Vector3(),
                      material=Au)]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
    geometry=geometry,
    default_material=mp.Medium(index=1.33)
)

box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

sim.load_minus_flux_data(box_x1, box_x1_data)
sim.load_minus_flux_data(box_x2, box_x2_data)
sim.load_minus_flux_data(box_y1, box_y1_data)
sim.load_minus_flux_data(box_y2, box_y2_data)
sim.load_minus_flux_data(box_z1, box_z1_data)
sim.load_minus_flux_data(box_z2, box_z2_data)

sim.run(until_after_sources=50)

box_x1_flux = mp.get_fluxes(box_x1)
box_x2_flux = mp.get_fluxes(box_x2)
box_y1_flux = mp.get_fluxes(box_y1)
box_y2_flux = mp.get_fluxes(box_y2)
box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)

scatt_flux = (
    np.asarray(box_x1_flux)
    - np.asarray(box_x2_flux)
    + np.asarray(box_y1_flux)
    - np.asarray(box_y2_flux)
    + np.asarray(box_z1_flux)
    - np.asarray(box_z2_flux)
)

intensity = np.asarray(box_x1_flux0)/(2*r)**2
scatt_cross_section = np.divide(scatt_flux, intensity)
scatt_eff = scatt_cross_section*(-1)/(np.pi*r**2)

sim.reset_meep()

# ------- with sphere (ABS) ---------

geometry = [mp.Sphere(radius=r,
                      center=mp.Vector3(),
                      material=Au)]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    symmetries=symmetries,
    geometry=geometry,
    default_material=mp.Medium(index=1.33)
)

box_x1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(frq_cen, dfrq, nfrq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

sim.run(until_after_sources=50)

box_x1_flux = mp.get_fluxes(box_x1)
box_x2_flux = mp.get_fluxes(box_x2)
box_y1_flux = mp.get_fluxes(box_y1)
box_y2_flux = mp.get_fluxes(box_y2)
box_z1_flux = mp.get_fluxes(box_z1)
box_z2_flux = mp.get_fluxes(box_z2)

abs_flux = (
    np.asarray(box_x1_flux)
    - np.asarray(box_x2_flux)
    + np.asarray(box_y1_flux)
    - np.asarray(box_y2_flux)
    + np.asarray(box_z1_flux)
    - np.asarray(box_z2_flux)
)

intensity = np.asarray(box_x1_flux0)/(2*r)**2
abs_cross_section = np.divide(abs_flux, intensity)
abs_eff = abs_cross_section/(np.pi*r**2)

wavelengths = 1/np.asarray(freqs)

# Stack the arrays horizontally
combined_array = np.column_stack((wavelengths, abs_eff, scatt_eff))

# Specify the filename
output_filename = 'output_arrays.txt'

# Save the combined array to a text file
np.savetxt(output_filename, combined_array, delimiter='\t', header='Wavelengths\tAbs\tScatt', comments='')

print(f"Arrays saved to {output_filename}")