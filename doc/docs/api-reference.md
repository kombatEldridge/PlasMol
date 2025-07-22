# API Reference

This page documents key classes and methods.

## Directory Tree

```
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
|  |  |
|  |  |_ propagators
|  |     |_ __init__.py
|  |     |_ magnus2.py
|  |     |_ rk4.py
|  |     |_ step.py
|  |
|  |_ utils
|  |  |_ __init__.py
|  |  |_ csv.py
|  |   |_ fourier.py
|  |   |_ gif.py
|  |   |_ logging.py
|  |   |_ plotting.py
|
|_ templates
   |_ template-classical.in
   |_ template-plasmol.in
   |_ template-quantum.in
```

## quantum/

### `molecule.py`

- Initializes molecule with PySCF, handles SCF, Fock matrix, dipole calculations.
- Additional methods can be injected at the bottom of this class to track other quantum properties. For example, we track the induced dipole using the `calculate_mu()` method and then inject the call to this method in the `quantum/propagation.py` file.

### `electric_field.py`

- Builds electric fields (pulse or kick shapes) for the molecule to feel *only* when a quantum simulation is chosen. If classical or full PlasMol simulation is running, the electric field is generated inside the Meep simulation (using `classical/sources.py`).
- Other electric field shapes for quantum simulations can be added to the `build_field()` method using the other two shapes as templates.

## quantum/propagators/

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

## classical/

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
        - where $\theta$ is the Heaviside step function, $t_{\text {start}}$ is the start time (default 0 ), and $\omega=2 \pi f$ with $f$ being the specified frequency.
    - `GaussianSource`
        - The `GaussianSource` generates a pulsed electromagnetic source with a Gaussian temporal envelope modulating a carrier wave, designed for broadband frequency excitation while minimizing truncation effects through a shifted peak.
        $$ 
            s(t)=\exp \left(-\frac{\left(t-t_0\right)^2}{2 w^2}\right) \exp (-i \omega t)
        $$
        - where $w$ is the width parameter, $t_0=t_{\text {start}}+c \cdot w$ with $c$ being the cutoff (default 5.0) to shift the peak and avoid abrupt truncation at $t=t_{\text {start}}$, and $\omega=2 \pi f$ with $f$ being the center frequency.
    - `ChirpedSource`
        - The `ChirpedSource` is a custom pulsed source featuring a Gaussian envelope with an added quadratic phase term to produce a linearly chirped frequency sweep, useful for applications requiring timevarying frequency content.
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

- Other electric field shapes for quantum simulations can be supported using the other two shapes as templates, but additional options for the sources will need to be added to the input file parser in the `input/params.py` file under the `getSource()` method (found in the `buildclassicalParams()` method).

## drivers/

### `classical.py`
- Runs Meep simulation.

### `quantum.py`
- Runs RT-TDDFT, supports multi-threading for transforms.

### `plasmol.py`
- Hybrid run.


## utils/

- `utils.csv.initCSV(filename, comment)`, `updateCSV(...)`.
- `utils.plotting.show_eField_pField(eFieldFile, pFieldFile)`.
- `utils.fourier.transform(...)`: Fourier transform and absorption spectrum.

[TODO: Add full parameter lists, return types, or use Sphinx for auto-gen docs.]
