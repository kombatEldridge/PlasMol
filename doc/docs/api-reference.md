# API Reference

This page documents key classes and methods, especially portions that can support custom injections for user's complete control over the simulation. If your changes to the code are a good fit for public use, feel free to read the [contributing](contributing.md) page.

## Directory Tree

```txt
PlasMol
|
|_ docs
|  |_ *files for these docs*
|  
|_ src
|  |_ __init__.py
|  |_ constants.py
|  |_ main.py
|  |
|  |_ classical
|  |  |_ __init__.py
|  |  |_ simulation.py
|  |  |_ sources.py
|  |
|  |_ drivers
|  |  |_ __init__.py
|  |  |_ classical.py
|  |  |_ plasmol.py
|  |  |_ quantum.py
|  |
|  |_ input
|  |  |_ __init__.py
|  |  |_ cli.py
|  |  |_ params.py
|  |  |_ parser.py
|  |
|  |_ quantum
|  |  |_ __init__.py
|  |  |_ chkfile.py
|  |  |_ electric_field.py
|  |  |_ molecule.py
|  |  |_ propagation.py
|  |  |_ propagators
|  |     |_ __init__.py
|  |     |_ magnus2.py
|  |     |_ rk4.py
|  |     |_ step.py
|  |
|  |_ utils
|  |  |_ __init__.py
|  |  |_ csv.py
|  |  |_ fourier.py
|  |  |_ gif.py
|  |  |_ logging.py
|  |  |_ plotting.py
|
|_ templates 
   |_ *files for Tutorial section*
```

## classical/

This directory contains all files necessary to simulate a nanoparticle in PlasMol. Included in it is the main file that runs both the classical and full PlasMol simulations (`simulation.py`) as well as the file used to build the electric fields in the simulations from input parameters (`sources.py`).

### `simulation.py`

- Runs Meep and full PlasMol simulation, handles sources, PML, symmetries.
- For those who want to add tracking to certain parameters, a commented section can be found in the `run()` method and right above it.
- An example code block is added (but commented out) to the `run()` method to graph the 3D model of the NP.

### `sources.py`

- Currently supports
    - `ContinuousSource`
        - The `ContinuousSource` provides a continuous-wave electromagnetic source with constant frequency and amplitude that activates abruptly at the start time and persists until the optional end time.
        $$
            s(t)=\theta\left(t-t_{\text {start}}\right) \exp (-i \omega t)
        $$
        - where $\theta$ is the Heaviside step function, $t_{\text {start}}$ is the start time (default 0), and $\omega=2 \pi f$ with $f$ being the specified frequency.
    - `GaussianSource`
        - The `GaussianSource` generates a pulsed electromagnetic source with a Gaussian temporal envelope modulating a carrier wave, designed for broadband frequency excitation while minimizing truncation effects through a shifted peak.
        $$
            s(t)=\exp \left(-\frac{\left(t-t_0\right)^2}{2 w^2}\right) \exp (-i \omega t)
        $$
        - where $w$ is the width parameter, $t_0=t_{\text {start}}+c \cdot w$ with $c$ being the cutoff (default 5.0) to shift the peak and avoid abrupt truncation at $t=t_{\text {start}}$, and $\omega=2 \pi f$ with $f$ being the center frequency.
    - `ChirpedSource`
        - The `ChirpedSource` is a custom pulsed source featuring a Gaussian envelope with an added quadratic phase term to produce a linearly chirped frequency sweep, useful for applications requiring time varying frequency content.
        $$
            s(t)=\exp \left(i 2 \pi f\left(t-t_p\right)\right) \exp \left(-a\left(t-t_p\right)^2+i b\left(t-t_p\right)^2\right)
        $$
        - where $f$ is the base frequency, $t_p$ is the peak time, $a$ is the width parameter controlling the envelope decay, and $b$ is the chirp rate introducing quadratic phase for frequency modulation.
    - `PulseSource`
        - When simulating an absorption spectrum, this source is required.
        - The `PulseSource` creates a custom Gaussian-enveloped pulse with a sinusoidal carrier wave peaked at a specified time, suitable for simulating short electromagnetic bursts in time-domain simulations.
        $$
            s(t)=\exp \left(i 2 \pi f\left(t-t_p\right)\right) \exp \left(-a\left(t-t_p\right)^2\right)
        $$
        - where $f$ is the carrier frequency, $t_p$ is the peak time, and $a$ is the width parameter determining the pulse duration.

- Other electric field shapes for the classical and full PlasMol simulations can be supported using these four as templates, but additional options for the sources will need to be added to the input file parser in the `input/params.py` file under the `getSource()` method (found in the `buildclassicalParams()` method).

## drivers/

For each type of simulation running, a surface-level file determines what steps to take to achieve the desired results. The reasoning for each type of simulation can be found on the [Usage](usage.md) page.

### `classical.py`

- Is chosen to run a MEEP simulation when only classical parameters are specified in the input file.
- A classical run has the capability to construct and run a MEEP simulation, as well as produce 2D cross-sections of the simulation at the center of the NP if the `hdf5` block is added to the input file.
- As previous discussed, other methods can be injected in the `classical/simulation.py` file to track other properties (like the commented out `getElectricField()` method which can track the electric field intensity at any pre-determined, hard-coded point).

### `quantum.py`

- Is chosen to run an RT-TDDFT simulation when only quantum parameters are specified in the input file.
- A quantum run by default will perform a single RT-TDDFT simulation given a molecule and an incident electric field.
- However, if `transform` is specified in the `rttddft` block of the input file, the simulation should be given an ultrafast pulse to perform a real time absorption spectrum calculation with three multithreaded simulations, one for each directional Gaussian pulse, and then a Fourier transform is performed to give the absorption spectrum (see [Tutorial #3](tutorials.md#tutorial-3-molecular-absorption-spectrum-rt-tddft-with-transform-flag)).

### `plasmol.py`

- Is chosen to run a full PlasMol simulation when both classical and quantum parameters are specified in the input file.
- This is the main purpose of PlasMol. A Meep simulation will begin with a molecule inside, whose initial electronic structure is built by PySCF. Every time step, the electric field at the molecule's position is measured and sent to the "quantum" portion of the code where the density matrix is propagated by the electric field. As an end result, the induced dipole moment of the molecule can be calculated. Finally, the induced dipole moment is fed back into the Meep simulation as the intensity of a point dipole at the position of the molecule.

## input/

Users should mainly interact with PlasMol through command line, calling upon the `main.py` script. To feed parameters into the simulation, users only have a few command line options (controlled by `cli.py`) and one input file (initially parsed by `parser.py` and then fit for the PlasMol codebase using `params.py`).

### `cli.py`

- Enumerates command line flags to be included when running PlasMol.
- Command line options include
    - `--input` (or `-f` for file): Path to the PlasMol input file.
    - `--log` (or `-l`): Path to the log file. If not specified, log prints to terminal and is not saved.
    - `--verbose` (or `-v` and `-vv`): Specifies verbosity levels.
        - Not specified: logs only `logger.warning` calls.
        - `-v`: logs up to `logger.info` calls.
        - `-vv`: logs up to `logger.debug` calls.
    - `--restart` (or `-r`): Tries to remove the following files from current working directory in case they are leftover from previous runs. Actual file names for these can be specified in the input file.
        - eField_path
        - pField_path
        - chkfile path
        - pField_Transform_path
        - eField_vs_pField_path
        - eV_spectrum_path

### `parser.py`

- Reads input file to determine what type of simulation is needed (classical, quantum, PlasMol).
- If needed, users should add new input line support to this file, as well as `params.py`.

### `params.py`

- After being told what type of simulation to run, this file prepares the `PARAMS` class which holds all necessary parameters for the simulations.
- If needed, users should add new input line support to this file, as well as `parser.py`.

## quantum/

This directory contains all files necessary to simulate a molecule in PlasMol. Included in it is the file to build the molecule (`molecule.py`), the file to build the electric field for just a quantum simulation (`electric_field.py`), the file to continually update the checkpoint files if called for (`chkfile.py`), and the files for propagating the density matrix of the molecule (`propagation.py` and `propagators/*`).

### `molecule.py`

- Initializes molecule with PySCF, handles SCF, Fock matrix, dipole calculations.
- Additional methods can be injected at the bottom of this class to track other quantum properties. For example, we track the induced dipole using the `calculate_mu()` method and then inject the call to this method in the `propagation.py` file.

### `electric_field.py`

- Builds electric fields (pulse or kick shapes) for the molecule to feel *only* when a quantum simulation is chosen. If classical or full PlasMol simulation is running, the electric field is generated inside the Meep simulation (using `classical/sources.py`).
- Other electric field shapes for quantum simulations can be added to the `build_field()` method using the other two shapes as templates.

### `propagation.py`

- Is called every time step, either by the quantum simulation in `drivers/quantum.py` or during the classical or full PlasMol simulations in `classical/simulation.py` inside `callQuantum()`.
- Additional methods can be injected into `molecule.py` to track other quantum properties and called at the bottom of this file.

### `chkfile.py`

- Save a checkpoint file containing the current state of the simulation, as specified by the chkfile frequency in the input file.
- Always saves the timestamp $t$, ground state density matrix $\mathbf{D}_{\text{AO}}(0)$, and the coefficient matrix at the timestamp $\mathbf{C}(t)$.
- Additionally, it will save
    - the orthogonalized coefficient matrix at the previous timestamp $\mathbf{C}_{\text{ortho}}(t-\Delta t)$ for the Magnus step propagator.
    - the orthogonalized Fock matrix at the previous half timestamp $\mathbf{F}_{\text{ortho}}(t-\frac{1}{2}\Delta t)$ for the 2nd order Magnus propagator.
- If a new propagator is added, any necessary quantities will need to be added to this file.

## quantum/propagators/

To complete the main loop of RT-TDDFT, the density matrix (here propagated using the coefficient matrix) should be propagated under some incident field. PlasMol has three propagators available. The Magnus step method (`step.py`) and the Runge-Kutta 4 (`rk4.py`) methods both do a fairly poor job at propagating at small time steps, but are given here as a comparative tool to the superior 2nd order Magnus (with a Predictor-Corrector loop) method (`Magnus2.py`). Though the 2nd order Magnus method is the field's standard, shorter and lower resolution simulations can be run for debugging using the former two methods.

### `step.py`

- Propagates molecular orbitals using the Magnus step method.
- This method is also known as the modified midpoint unitary transformation (MMUT) scheme from [https://doi.org/10.1039/B415849K](https://doi.org/10.1039/B415849K )

$$
    \begin{gathered}
    \mathbf{F}_{\text {orth }}=\text {molecule.get_F_orth}\left( \mathbf{D}_{\text {AO }}, \vec{\mathbf{E}}\right)\\
    \mathbf{U} =\exp \left(-2 i \Delta t \cdot \mathbf{F}_{\text {orth}} (t)\right)\\
    \mathbf{C}_{\text{ortho}} \left(t + \Delta t\right)= \mathbf{U} \mathbf{C}_{\text{ortho}} \left(t - \Delta t\right)
    \end{gathered}
$$

### `rk4.py`

- Propagates molecular orbitals using the Runge-Kutta 4 method.

$$
    \begin{gathered}
    \mathbf{F}_{\text {orth }}=\text {molecule.get_F_orth}\left( \mathbf{D}_{\text {AO }}, \vec{\mathbf{E}}\right)\\
    k _1=-i \Delta t \cdot \mathbf{F}_{\text {orth }} \mathbf{C}_{\text {orth }} \\
    \mathbf{C}_{\text {orth, } 1}= \mathbf{C}_{\text {orth }}+\frac{1}{2} k _1 \\
    k _2=-i \Delta t \cdot \mathbf{F}_{\text {orth }} \mathbf{C}_{\text {orth, } 1} \\
    \mathbf{C}_{\text {orth, } 2}= \mathbf{C}_{\text {orth }}+\frac{1}{2} k _2 \\
    k _3=-i \Delta t \cdot \mathbf{F}_{\text {orth }} \mathbf{C}_{\text {orth, } 2} \\
    \mathbf{C}_{\text {orth, } 3}= \mathbf{C}_{\text {orth }}+ k _3 \\
    k _4=-i \Delta t \cdot \mathbf{F}_{\text {orth }} \mathbf{C}_{\text {orth, } 3} \\
    \mathbf{C}_{\text {orth }}(t+\Delta t)= \mathbf{C}_{\text {orth }}+\frac{ k _1}{6}+\frac{ k _2}{3}+\frac{ k _3}{3}+\frac{ k _4}{6}
    \end{gathered}
$$

### `magnus2.py`

- Propagates molecular orbitals using the second order Magnus method with a predictor-corrector algorithm included.
- This method is described in many RT-TDDFT papers, but I found the derivation from [https://doi.org/10.1021/ct501078d](https://doi.org/10.1021/ct501078d) to be very understandable.

$$
    \mathbf{F}_{\text{orth }}^{(0)}(t+\Delta t / 2)=2 \mathbf{F}_{\text{orth }}(t)- \mathbf{F}_{\text{orth }}(t-\Delta t / 2)
$$

- Then, for iterations $k=0,1,2, \ldots$ until convergence or maximum iterations exceeded:

$$
    \begin{gathered}
    \mathbf{U}^{(k)}=\exp \left(-i \Delta t \cdot \mathbf{F}_{\text {orth }}^{(k)}(t+\Delta t / 2)\right) \\
    \mathbf{C}_{\text {orth }}^{(k+1)}(t+\Delta t)= \mathbf{U}^{(k)} \mathbf{C}_{\text {orth }}(t) \\
    \mathbf{C}^{(k+1)}(t+\Delta t)=\text {molecule.rotate_coeff_away_from_orth}\left( \mathbf{C}_{\text {orth }}^{(k+1)}(t+\Delta t)\right) \\
    \mathbf{D}_{\text {AO }}^{(k+1)}(t+\Delta t)=\text {molecule.make_rdm1}\left( \mathbf{C}^{(k+1)}(t+\Delta t), \text { occ }\right) \\
    \mathbf{F}_{\text {orth }}^{(k+1)}(t+\Delta t)=\text {molecule.get_F_orth}\left( \mathbf{D}_{\text {AO }}^{(k+1)}(t+\Delta t), \vec{\mathbf{E}}\right)
    \end{gathered}
$$

- Update the midpoint guess for the next iteration:

$$
    \mathbf{F}_{\text {orth }}^{(k+1)}(t+\Delta t / 2)=\frac{1}{2}\left( \mathbf{F}_{\text {orth }}(t)+ \mathbf{F}_{\text {orth }}^{(k+1)}(t+\Delta t)\right)
$$

- Convergence is checked starting from the second iteration ($k \geq 1$) as:

$$
    \left\| \mathbf{C}^{(k+1)}(t+\Delta t)- \mathbf{C}^{(k)}(t+\Delta t)\right\|<\epsilon
$$

- where $\epsilon=$ `params.pc_convergence`. Upon convergence, assign the final values to the molecule object:

$$
    \begin{gathered}
    \mathbf{C}(t+\Delta t)= \mathbf{C}^{(\text {final})}(t+\Delta t), \quad \mathbf{D} _{\text{AO}}(t+\Delta t)= \mathbf{D} _{\text{AO}}^{(\text {final})}(t+\Delta t) \\
    \mathbf{F}_{\text {orth }}(t+\Delta t)= \mathbf{F}_{\text {orth }}^{(\text {final})}(t+\Delta t), \quad \mathbf{F}_{\text {orth }}(t+\Delta t / 2)= \mathbf{F}_{\text {orth }}^{(\text {final})}(t+\Delta t / 2)
    \end{gathered}
$$

The updated $\mathbf{F}_{\text{orth }}(t+\Delta t / 2)$ is stored for extrapolation in the next time step.

## utils/

This directory stores files that add utilities to the PlasMol codebase, but are too large or cumbersome to be placed inside other files. Such files include the handler for reading and writing to csv files (`csv.py`), the handler for creating a gif from the generated hdf5 files (`gif.py`), the handler for logging (`logging.py`), the handler for plotting data found in csv files (`plotting.py`), and the methods for Fourier transforming the data for the absorption spectra (`fourier.py`).
