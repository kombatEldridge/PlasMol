import numpy as np
import pandas as pd

def read_dipole_component(filename):
    df = pd.read_csv(filename, delimiter=',', skiprows=1, header=None, names=['time', 'dipole'], comment='#')
    
    # Remove any rows with missing or malformed data
    df = df.dropna()
    
    # Extract columns and convert to NumPy arrays explicitly
    time = np.array(df['time'].values)
    dipole = np.array(df['dipole'].values)
    
    # Return as NumPy arrays
    return np.array(time), np.array(dipole)

# Load components
tx, dx = read_dipole_component("xx.csv")
ty, dy = read_dipole_component("yy.csv")
tz, dz = read_dipole_component("zz.csv")

# Find the length of the shortest file
min_length = min(len(tx), len(ty), len(tz))

# Trim all arrays to the shortest length
tx = tx[:min_length]
ty = ty[:min_length]
tz = tz[:min_length]
dx = dx[:min_length]
dy = dy[:min_length]
dz = dz[:min_length]

# Check consistency
if not (np.allclose(tx, ty) and np.allclose(tx, tz)):
    raise ValueError("Time points do not match across files!")

# Build time_points and dipole_moment array
time_points = tx
dipole_moment = np.vstack([dx, dy, dz])  # Shape: (3, N)

# Example metadata
basis = "6-31G"
dt = time_points[1] - time_points[0]
field_strength = 5e-5
method = "DFT"
HOMO = -0.33331

# Save to .npz
np.savez("h2o.npz",
         basis=basis,
         dt=dt,
         dipole_moment=dipole_moment,
         field_strength=field_strength,
         method=method,
         HOMO=HOMO,
         time_points=time_points)
