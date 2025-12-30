# quantum/electric_field.py
import logging 
import numpy as np

from .. import constants

logger = logging.getLogger("main")

class ELECTRICFIELD:
    def __init__(self, times, params):
        self.shape = params.shape
        self.intensity_au = params.intensity_au
        self.peak_time_au = params.peak_time_au
        self.width_au = params.width_steps * params.dt
        self.times = times
        self.dir = params.dir

        if self.shape == 'pulse':
            self.wavelength_nm = params.wavelength_nm
            self.wavelength_au = self.wavelength_nm * constants.D_AU_NM
        elif self.shape == 'kick':
            pass
        # elif self.shape == 'custom_shape'...
        #   if additional parameters need to be added for your custom shape
        #   you must add support for them in the input/params.py method: buildQuantumParams()
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
            omega = 2 * np.pi * constants.C_AU / self.wavelength_au
            carrier  = np.exp(1j * omega * (t - self.peak_time_au))
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = self.intensity_au * np.real(carrier * envelope)
        elif self.shape == 'kick':
            envelope = np.exp(-((t - self.peak_time_au)**2) / (2 * self.width_au**2))
            active_component = np.zeros_like(t)  # Initialize array of zeros matching t's shape and dtype
            mask = np.isclose(t, self.peak_time_au, atol=1e-10)  # Adjust atol for your time precision; finds matching indices
            if np.any(mask):  # Optional: Check if there's at least one match
                active_component[mask] = self.intensity_au
            else:
                # Optional: Handle no exact match, e.g., find closest index
                idx = np.argmin(np.abs(t - self.peak_time_au))
                active_component[idx] = self.intensity_au            # * envelope
        # ------------------------------------ #
        #              Additional              #
        #    custom sources can be defined     #
        #      here and supported above        #
        #             as commented             #
        # ------------------------------------ #


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

