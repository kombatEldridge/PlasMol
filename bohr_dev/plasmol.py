import os
import numpy as np
import meep as mp
import sys
from gif import make_gif
from meep.materials import Au_JC_visible as Au
import bohr
from scipy import constants

# Location of molecule's input file in Psi4 format
inputfile = sys.argv[1]
convertTimeMeeptoSI = 10 / 3  # t_Meep * convertTimeMeeptoSI = t_SI
convertTimeBohrtoSI = 0.024188843  # t_Bohr * convertTimeBohrtoSI = t_SI
convertFieldMeeptoSI = 1 / 1e-6 / constants.epsilon_0 / constants.c / 0.51422082e12
responseCutOff = 1e-12

# -------- MEEP INIT -------- #

# Nanoparticle's radius in microns (the base unit in meep)
radiusNP = 0.025

# Define wavelength range for the simulation
wavelengthMin = 0.400
wavelengthMax = 0.800
frequencyMin = 1/wavelengthMax  # Minimum frequency
frequencyMax = 1/wavelengthMin  # Maximum frequency
frequencyCenter = 0.5*(frequencyMin+frequencyMax)  # Center frequency
frequencyWidth = frequencyMax-frequencyMin  # Frequency width
frequencySamplePts = 50  # Number of frequency points
frequencyRange = mp.FreqRange(
    min=frequencyMin, max=frequencyMax)  # Frequency range for MEEP

resolution = 1000  # Spatial resolution of the simulation

# Thickness of Perfectly Matched Layer (PML) absorbing boundaries
pmlThickness = 0.01
pmlList = [mp.PML(thickness=pmlThickness)]  # Define PML list

# Define symmetry conditions for the simulation
symmetriesList = [mp.Mirror(mp.Y), mp.Mirror(mp.Z, phase=-1)]

cellLength = 0.1  # Size of the simulation cell
# Set the dimensions of the simulation cell
cellVolume = mp.Vector3(cellLength, cellLength, cellLength)

# Position of nanoparticle and molecule
positionNP = mp.Vector3(0, 0, 0)  # Nanoparticle at origin
# Molecule 0.010 microns off NP surface in the x direction
positionMolecule = mp.Vector3(radiusNP + 0.010, 0, 0)

# Define the frame coordinates for nanoparticle and molecule
frameCenter = cellLength * resolution / 2
frameMoleculeX = (positionMolecule.x + cellLength / 2) * resolution
frameMoleculeY = (positionMolecule.y + cellLength / 2) * resolution
frameMoleculeZ = (positionMolecule.z + cellLength / 2) * resolution
frameNPX = (positionNP.x + cellLength / 2) * resolution
frameNPY = (positionNP.y + cellLength / 2) * resolution
frameNPZ = (positionNP.z + cellLength / 2) * resolution

# These functions get called every half time step (FDTD nuance) so dipoleResponse must have values for t and t+dt/2
def chirpx(t):
    return dipoleResponse['x'][str(round(t, decimalPlaces))] if dipoleResponse['x'][str(round(t, decimalPlaces))] != 0 else 0


def chirpy(t):
    return dipoleResponse['y'][str(round(t, decimalPlaces))] if dipoleResponse['y'][str(round(t, decimalPlaces))] != 0 else 0


def chirpz(t):
    return dipoleResponse['z'][str(round(t, decimalPlaces))] if dipoleResponse['z'][str(round(t, decimalPlaces))] != 0 else 0


# Define sources for the simulation
sourcesList = [
    mp.Source(
        # Frequency 100 corresponds to 100nm wavelength
        mp.ContinuousSource(frequency=100, is_integrated=True),
        center=mp.Vector3(-0.5 * cellLength + pmlThickness),
        size=mp.Vector3(0, cellLength, cellLength),
        component=mp.Ez
    ),
    # mp.Source(
    #     mp.GaussianSource(frq_cen, fwidth=dfrq, is_integrated=True),
    #     center=mp.Vector3(-0.5*s+dpml),
    #     size=mp.Vector3(0, s, s),
    #     component=mp.Ez,
    # ),
    mp.Source(
        mp.CustomSource(src_func=chirpx),
        center=positionMolecule,
        component=mp.Ex,
    ),
    mp.Source(
        mp.CustomSource(src_func=chirpy),
        center=positionMolecule,
        component=mp.Ey
    ),
    mp.Source(
        mp.CustomSource(src_func=chirpz),
        center=positionMolecule,
        component=mp.Ez
    )
]

# Define the geometry of the simulation (a gold sphere as the nanoparticle)
objectList = [mp.Sphere(radius=radiusNP, center=positionNP, material=Au)]

# Initialize the simulation object
sim = mp.Simulation(
    resolution=resolution,
    cell_size=cellVolume,
    boundary_layers=pmlList,
    sources=sourcesList,
    symmetries=symmetriesList,
    geometry=objectList,
    default_material=mp.Medium(index=1.33)
)

# Define the time step for the simulation
timeStepMeep = sim.Courant / sim.resolution  # One time unit in MEEP is 3.33 fs
timeStepBohr = 2 * timeStepMeep * convertTimeMeeptoSI / convertTimeBohrtoSI

initialDipole = 0  # Initial dipole moment

# Initialize dipole array as a nested dictionary
dipoleResponse = {'x': {}, 'y': {}, 'z': {}}
indexForComponents = {0: 'x', 1: 'y', 2: 'z'}
for i in np.arange(0, 3):
    componentName = indexForComponents[i]
    dipoleResponse[componentName] = {}
    dipoleResponse[componentName][str(0 * timeStepMeep)] = initialDipole
    dipoleResponse[componentName][str(0.5 * timeStepMeep)] = initialDipole
    dipoleResponse[componentName][str(1 * timeStepMeep)] = initialDipole

# Determine the number of decimal places for time steps
# String representation of half the time step
halfTimeStepString = str(timeStepMeep / 2)
# Position of the decimal point in the string
decimalPointIndex = halfTimeStepString.find('.')
decimalPlaces = len(halfTimeStepString) - \
    decimalPointIndex - 1  # Number of decimal places

# Initialize arrays to store electric field components
electricFieldArray = {'x': [], 'y': [], 'z': []}
fieldComponents = [mp.Ex, mp.Ey, mp.Ez]

# Function to get a slice of the electric field at each time step


def getElectricField(sim):
    global electricFieldArray
    print("Getting Electric Field at the molecule at time ",
          sim.meep_time()*convertTimeMeeptoSI, " fs")
    for i in np.arange(0, 3):
        componentName = indexForComponents[i]
        electricFieldArray[componentName].append(np.mean(sim.get_array(
            component=fieldComponents[i], center=positionMolecule, size=mp.Vector3(1E-20, 1E-20, 1E-20)))*convertFieldMeeptoSI)
        dipoleResponse[componentName][str(
            round(sim.meep_time() + (0.5 * timeStepMeep), decimalPlaces))] = initialDipole
        dipoleResponse[componentName][str(
            round(sim.meep_time() + timeStepMeep, decimalPlaces))] = initialDipole
    return 0


def callBohr(sim):
    global electricFieldArray
    moleculeResponse = {'x': [0], 'y': [0], 'z': [0]}
    print("Calling Bohr for molecule's response at time ",
          sim.meep_time()*convertTimeMeeptoSI, " fs")
    for i in np.arange(0, 3):
        componentName = indexForComponents[i]
        if (np.mean(electricFieldArray[componentName]) < responseCutOff):
            moleculeResponse[componentName] = 0
        else:
            moleculeResponse[componentName] = bohr.run(
                inputfile, electricFieldArray['x'], electricFieldArray['y'], electricFieldArray['z'], timeStepBohr)[i]
        dipoleResponse[componentName][str(round(
            sim.meep_time() + (0.5 * timeStepMeep), decimalPlaces))] = moleculeResponse[componentName]
        dipoleResponse[componentName][str(
            round(sim.meep_time() + timeStepMeep, decimalPlaces))] = moleculeResponse[componentName]

    print("\t Molecule's Response: ", moleculeResponse, "\n")
    # Reset electricFieldArray for the next iteration
    electricFieldArray = {'x': [], 'y': [], 'z': []}
    return 0


# Set the output directory for the simulation results
script_name = "testingGifOutput"
sim.use_output_directory(script_name)

# Function to clear all files in the specified directory
def clear_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file_name in files:
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        print(f"All files in {directory_path} have been deleted.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Clear the output directory before starting the simulation
clear_directory(script_name)

# For color scaling in the output (TODO: automate this)
intensityMin = -3
intensityMax = 3

# Run the simulation, defining actions at specific intervals
sim.run(
    # Call getSlice at every time step dT
    mp.at_every(timeStepMeep, getElectricField),
    # Call callBohr every 3 time steps
    mp.at_every(3 * timeStepMeep, callBohr),
    # Output images every 10 time steps
    mp.at_every(10 * timeStepMeep, mp.output_png(mp.Ez,
                f"-X 10 -Y 10 -m {intensityMin} -M {intensityMax} -z {frameCenter} -Zc dkbluered")),
    until=500 * timeStepMeep  # Run the simulation until 300 time steps
)

# Create a GIF from the output images
make_gif(script_name)
