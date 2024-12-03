# PlasMol: Plasmon-Molecule Interaction Simulation

PlasMol is a Python-based simulation framework that combines classical and quantum mechanics methods to model the interaction between plasmonic nanoparticles and molecules. It calculates the induced dipole moment of a molecule subjected to an electric field generated by a plasmonic nanoparticle using a combination of time-dependent density functional theory (TDDFT) and finite-difference time-domain (FDTD) methods.

> :warning: This README is not currently accurate but I'm keeping it here for future use.

## Key Features

* **Coupled QM and Meep Simulation:**  Combines quantum mechanical calculations (currently using PySCF and RK4) with electromagnetic simulations (Meep) for a comprehensive analysis of plasmon-molecule interactions.
* **Support for Different Source Types:** Allows for the use of continuous, Gaussian, and chirped sources.
* **Configurable Simulation Parameters:** Offers flexibility in setting various simulation parameters such as cell size, resolution, and runtime.
* **Output Generation:** Produces CSV files with the electric and polarization fields and image files visualizing the electric field during the simulation, as well as creating a GIF animation of the simulation's evolution.


## Technologies Used

* **Python:** The primary programming language.
* **PySCF:**  A Python-based quantum chemistry package for electronic structure calculations.
* **[Meep](https://github.com/nanocomp/meep):** An open-source software package for electromagnetic simulations using the FDTD method from MIT.
* **NumPy:**  For numerical computations.
* **Pandas:** For data manipulation and analysis.
* **Matplotlib:** For creating visualizations.
* **Pillow (PIL):** For image manipulation and GIF creation.


## Prerequisites

Before running PlasMol, ensure you have the following installed:

* **Python 3.7 or higher:**  PlasMol is compatible with Python 3.7 and later.
* **PySCF:** Install using `pip install pyscf`.  You may need to install additional dependencies for PySCF depending on your chosen method. Check PySCF's documentation for details.
* **Meep:** Install according to the instructions on the Meep website ([https://meep.readthedocs.io/en/latest/](https://meep.readthedocs.io/en/latest/)).  Ensure Meep's python bindings are correctly configured.
* **NumPy:** Install using `pip install numpy`.
* **Pandas:** Install using `pip install pandas`.
* **Matplotlib:** Install using `pip install matplotlib`.
* **Pillow:** Install using `pip install Pillow`.

**Note:** The `create_meep_script.sh` script assumes you have a SLURM cluster environment set up, along with the necessary modules for meep and python. Adjust paths to match your specific setup.


## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Navigate to the project directory:**
   ```bash
   cd <project_directory>
   ```
3. **Install dependencies (if using pip):**
   ```bash
   pip install -r requirements.txt  
   ```
   Note that this project does not contain a `requirements.txt` file.  You should manually install the prerequisites listed above.

4. **Configure PySCF:**  The code uses PySCF. For example, you might need to modify  `/Users/bldrdge1/.conda/envs/meep/lib/python3.11/site-packages/pyscf/__config__.py` to set `B3LYP_WITH_VWN5 = True` (as noted in the code).  This path will be different on your system.

5. **Prepare Input Files:** You'll need two main input files: a Meep input file (`.in`) and a molecule input file (e.g., `pyridine.in`). The `create_meep_script.sh` script can help with generating a Meep input file based on a template.  The molecule file needs to be in the format expected by PySCF, usually a string defining atoms and coordinates, or a pathway to an existing geometry file in a format that PySCF supports.


## Usage

The main script is `driver.py`. To run the simulation:

```bash
python bohr_dev/driver.py -m <path_to_meep_input_file> -b <path_to_molecule_input_file> [-l <log_file_name>] [-v]
```
* `-m` or `--meep`: Path to the Meep input file (required).
* `-b` or `--bohr`: Path to the Bohr input file (required).  The example uses `pyridine.in`.
* `-l` or `--log`:  Name of the log file (optional).
* `-v`: Verbose mode (optional, add more `-v` flags for increased verbosity.  `-vv` is equivalent to `-v -v`).

The `create_meep_script.sh` script helps generate a Meep input file.  Run it to be guided through the necessary inputs.

**Example `pyridine.in` content (molecule-Files/molecule-template.in):**  This file needs to define the molecule's geometry and other relevant parameters for PySCF. The example shows a template; fill it accordingly.

**Example `meep.in` content (molecule-Files/meep-template.in):** This file defines simulation parameters, source characteristics and output settings. The example shows a template, use the `create_meep_script.sh` script to generate an appropriate file.


## Configuration

### Meep Input File (`.in`)

The Meep input file uses sections to define different parameters:

* **`source` section:** Specifies the type (`continuous`, `gaussian`, `chirped`), parameters (wavelength, frequency, width, etc.) and placement of the light source.  See `sources.py` for details.
* **`molecule` section:**  Defines the molecule's geometry (coordinates and atoms), parameters for PySCF calculations, and response directions.
* **`simulation` section:** Specifies simulation parameters like resolution, cell length, PML thickness, total simulation time, and symmetry conditions.
* **`outputPNG` section:** (Optional) Defines parameters for creating PNG images during the simulation, including the frequency of image generation, intensity range, and the output directory name.
* **`matplotlib` section:** (Optional) Specifies parameters to generate graphs using matplotlib.  The 'output' value in the matplotlib section will be used to create the filename.

### Bohr Input File (`pyridine.in`)

This file contains information about the molecule itself, specifically parameters for PySCF calculations such as:

* `charge`: Molecular charge.
* `spin`:  Spin multiplicity.
* `basis`:  Basis set to use (e.g., `6-31g`).
* `method`:  The computational method for TDDFT (e.g., `rttddft`).
* `resplimit`: Cutoff for induced dipole response magnitude.  The simulation will not run the Bohr calculation for dipole moments with magnitudes smaller than this value.
* Other PySCF options.

## Project Structure

```
bohr_dev/
├── basis/          # Directory containing basis sets for PySCF.
│   ├── modbas.2c    
│   ├── modbas.4c
│   ├── sapporo-dkh3-dzp-2012-diffuse
│   └── ...
├── bohr.py         # Contains the main Bohr dipole moment calculation functions.
├── driver.py       # Main script to run the simulation.
├── gif.py          # Functions for creating GIF animations.
├── plasmol.py      # Sets up and runs the Meep simulation.
├── simulation.py  # Contains the `Simulation` class.
├── sources.py      # Defines source classes (ContinuousSource, GaussianSource, ChirpedSource).
└── ...
molecule-Files/
├── meep-template.in # Template Meep input file.
├── molecule-template.in # Template molecule input file.
└── molecule-template.mos # Template of molecule guess_mos file
.gitignore         # Git ignore file
create_meep_script.sh # Script for creating Meep input files.
```

## Contributing

(No contributing guidelines found in the provided codebase.)


## License

(No license information found in the provided codebase.)


## Error Messages

* **`ValueError: Must provide either timeLength or totalTime with proper unit. Neither found.`:**  You must specify either `timeLength` or `totalTime` in your Meep input file's `simulation` section.
* **`ValueError: Unsupported source type: <source_type>`:** The specified `source_type` in your Meep input file is not supported.
* **`ValueError: Unsupported material type: <material>`:** The material specified in the `object` section of the Meep input file is not defined in `mp.materials`.
* **`ValueError: Directions should be x, y, and/or z separated by spaces '<value>'`:** The `directionCalculation` parameters in the Meep and molecule input files should contain only 'x', 'y', and 'z' values separated by spaces.
* **`ValueError: If you want to generate pictures, you must provide timestepsBetween, intensityMin, and intensityMax.`:** You must define the `timestepsBetween`, `intensityMin`, and `intensityMax` values in the `outputPNG` section of your Meep input file.
* **Other PySCF Errors:**  You may encounter errors related to PySCF calculations if your molecule input file is improperly formatted or the chosen computational method has problems. Refer to PySCF's documentation for troubleshooting.
* **Meep Errors:** Meep may generate various errors if simulation parameters are incorrect.  Consult Meep's documentation to resolve these.

