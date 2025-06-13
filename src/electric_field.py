# electric_field.py
import numpy as np
import logging 

logger = logging.getLogger("main")

C_NM_FS = 299.792458    # Speed of light in nm/fs
A0 = 5.29177210903e-11  # Bohr radius in m
D_AU_NM = 1e-9 / A0     # Distance conv in au/nm
D_NM_CM = 1e-7          # Distance conv in nm/cm
T_AU_FS = 41.3413733    # Time conv in au/fs
E_EV_AU = 27.21138602   # Energy conv in eV/au
E_NM_EV = 1239.84193    # Energy conv in eV nm
C_AU = 137.035999       # Speed of light in atomic units

class ELECTRICFIELD:
    def __init__(self, times, dir, wavelength_nm, peak_time_au,
                 width_steps, dt, shape, intensity_au):
        self.shape = shape
        self.intensity_au = intensity_au
        self.peak_time_au = peak_time_au
        self.width_au = width_steps * dt
        self.times = times
        self.dir = dir

        if shape == 'pulse':
            self.wavelength_nm = wavelength_nm
            self.wavelength_au = self.wavelength_nm * D_AU_NM
        elif shape == 'kick':
            pass
        else:
            raise ValueError("Invalid shape. Must be 'pulse' or 'kick'.")
        
        self.field = self.build_field()

    def build_field(self):
        """
        Compute the electric field for a given array of times.

        Generates the field based on the shape: 'pulse' (oscillatory) or 'kick' (Gaussian),
        applying it along the specified direction with optional smoothing.

        Parameters:
        times : np.ndarray
            Array of times in atomic units.
        dir : str
            Direction of the field ('x', 'y', or 'z').

        Returns:
        np.ndarray
            Array of shape (len(times), 3) with field components [x, y, z].
        """

        t = np.asarray(self.times) # must be in au
        if self.shape == 'pulse':
            omega = 2 * np.pi * C_AU / self.wavelength_au
            carrier  = np.exp(1j * omega * (t - self.peak_time_au))
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * np.real(carrier * envelope)
        elif self.shape == 'kick':
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * envelope

        field = np.zeros((len(t), 3))
        dir = self.dir.lower()
        if dir == 'x':
            field[:, 0] = active_component
        elif dir == 'y':
            field[:, 1] = active_component
        elif dir == 'z':
            field[:, 2] = active_component
        else:
            raise ValueError("Invalid direction. Must be 'x', 'y', or 'z'.")
        return field

