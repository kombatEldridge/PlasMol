# Tutorials

These tutorials demonstrate core workflows. Assume you've installed PlasMol.

## Tutorial 1: Simple Meep Simulation
1. Create `input.meep`:
   ```
start meep
simulation
  cellLength = 10
  resolution = 20
  pmlThickness = 1.0
end
source
  sourceType = 'gaussian'
  frequency = 0.5
  [etc.]
end
```
2. Run:
   ```bash
   python -m src.main -f input.meep
   ```
3. View outputs: Check eField.csv and GIF.

Expected: Simulates Gaussian pulse propagation.

## Tutorial 2: RT-TDDFT Quantum Simulation
1. Create `input.quantum`:
   ```
start quantum
  rttddft
    basis = 'sto-3g'
    xc = 'lda'
    geometry
      H 0 0 0.0 0.0
      O 1.0 0.0 0.0
    end geometry
  end
  propagator = 'rk4'
end
source
  shape = 'pulse'
  wavelength_nm = 400
  [etc.]
end
```
2. Run**:
   ```bash
  python -m src.main -f input.quantum -v
  ```
3. Analyze: Use plotting.py to visualize fields; apply fourier.transform for spectrum.

Expected: Propagates density matrix, outputs polarization and HOMO-LUMO jumps.

## Tutorial 3: Hybrid PlasMol
Combine Meep and quantum inputs. Run similarly.

[TODO: Add more detailed examples, code snippets, or Jupyter notebook links.]
